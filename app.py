import os, json, logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types
from collections import defaultdict, deque
from datetime import datetime

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- VALIDACIÓN AL ARRANCAR ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ FALTA GEMINI_API_KEY en el .env")

app = Flask(__name__)
CORS(app)
client = genai.Client(api_key=API_KEY)

# --- MEMORIA CONVERSACIONAL (en RAM, últimos 6 turnos por usuario) ---
memoria = defaultdict(lambda: deque(maxlen=6))

# --- CARGA DE DATOS ---
db_clientes = {}
try:
    with open('clientes_etiquetados.json', 'r', encoding='utf-8') as f:
        clientes_data = json.load(f)
        db_clientes = {str(c.get('user_id', '')): c for c in clientes_data if c.get('user_id')}
    logging.info(f"✅ Cerebro cargado: {len(db_clientes)} perfiles 360°")
except Exception as e:
    logging.error(f"❌ Error cargando JSON: {e}")


# --- 🧠 PRE-PROCESADOR DE INSIGHTS ---
# Aquí TÚ traduces los 80 campos crudos a insights legibles para Gemini
def construir_ficha_inteligente(c: dict) -> str:
    """Convierte el JSON crudo en un brief accionable para la IA."""
    
    # Señales clave
    alertas = []
    oportunidades = []
    
    # Riesgo crediticio
    if c.get('score_buro', 850) < 600:
        alertas.append(f"🚨 Score buró BAJO: {c.get('score_buro')} (riesgo impago)")
    
    # Utilización de crédito
    util = c.get('utilizacion_promedio', 0)
    if util > 0.7:
        alertas.append(f"🚨 Usa {util*100:.0f}% de su crédito (saturado)")
    elif util > 0.3:
        alertas.append(f"⚠️ Usa {util*100:.0f}% de su crédito")
    
    # Actividad reciente
    dias_inactivo = c.get('dias_desde_ultima_transaccion', 0)
    if dias_inactivo > 15:
        alertas.append(f"⚠️ {dias_inactivo} días sin transaccionar")
    
    # Variación actividad
    var = c.get('variacion_actividad', 0)
    if var < -0.3:
        alertas.append(f"📉 Actividad cayó {abs(var)*100:.0f}% vs mes pasado")
    
    # Oportunidades
    if not c.get('tiene_seguro'):
        oportunidades.append("Sin seguro contratado")
    if not c.get('nomina_domiciliada'):
        oportunidades.append("Nómina NO domiciliada")
    if c.get('nivel_oportunidad_transaccional') == 'Alto':
        oportunidades.append("Alto potencial comercial")
    if c.get('cashback_total', 0) < 200:
        oportunidades.append("Bajo aprovechamiento de cashback")
    
    ficha = f"""
PERFIL DEC: {c.get('perfil_negocio', 'N/D')}
DEMOGRAFÍA: {c.get('edad')} años, {c.get('sexo')}, {c.get('ciudad')}, ingreso ${c.get('ingreso_mensual_mxn'):,} MXN/mes
SATISFACCIÓN: {c.get('satisfaccion_1_10')}/10 | Antigüedad: {c.get('antiguedad_dias')} días
PRODUCTOS: {c.get('num_productos_activos')} activos (principal: {c.get('producto_principal')})
SALDO TOTAL: ${c.get('saldo_total_productos', 0):,.0f} | Límite crédito: ${c.get('limite_credito_total', 0):,.0f}
HÁBITOS: Canal favorito {c.get('canal_principal')}, gasta principalmente en {c.get('categoria_principal')}
INSIGHT IA: {c.get('insight_transaccional', 'N/D')}

🚨 ALERTAS: {' | '.join(alertas) if alertas else 'Ninguna'}
💡 OPORTUNIDADES: {' | '.join(oportunidades) if oportunidades else 'Ninguna'}
"""
    return ficha.strip()


# --- 🎭 SYSTEM INSTRUCTION ---
def construir_system_prompt(ficha: str) -> str:
    return f"""Eres HaviEr, la inteligencia proactiva y empática de Hey Banco.

REGLAS DE ORO:
1. NO te presentes, ve directo al grano.
2. Máximo 2-3 oraciones. Tono regio, cálido, futurista.
3. NO recites números crudos. Tradúcelos en consejos accionables.
4. Si detectas alertas, abórdalas con tacto (no asustes al cliente).
5. Sugiere productos solo si encajan con el perfil real.
6. Usa emojis con moderación (1 máximo por respuesta).
7. Habla de "tú" (no "usted").

FICHA 360° DEL CLIENTE:
{ficha}
"""


# --- ENDPOINTS ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "clientes_cargados": len(db_clientes)})


@app.route('/api/clientes', methods=['GET'])
def listar_clientes():
    """Útil para el frontend del demo."""
    return jsonify({
        "total": len(db_clientes),
        "ids": list(db_clientes.keys())[:50]
    })


@app.route('/api/insight-proactivo', methods=['POST'])
def insight_proactivo():
    """HaviEr habla PRIMERO al abrir el chat."""
    try:
        uid = str(request.json.get('id_cliente', ''))
        cliente = db_clientes.get(uid)
        
        if not cliente:
            return jsonify({"respuesta_havi": "¡Qué onda! Soy HaviEr. ¿En qué te ayudo hoy?"})
        
        ficha = construir_ficha_inteligente(cliente)
        system = construir_system_prompt(ficha)
        
        prompt_proactivo = (
            "Genera UN saludo proactivo y personalizado para este cliente. "
            "Detecta lo MÁS urgente o relevante de su perfil (alerta u oportunidad) "
            "y menciónalo en forma de pregunta amigable. Máximo 2 oraciones."
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt_proactivo,
            config=types.GenerateContentConfig(system_instruction=system)
        )
        
        texto = response.text or "¡Qué onda! ¿En qué te ayudo hoy?"
        
        # Guardamos en memoria
        memoria[uid].append({"rol": "havi", "msg": texto})
        
        return jsonify({"status": "success", "respuesta_havi": texto})
    
    except Exception as e:
        logging.error(f"⚠️ Error proactivo: {e}")
        return jsonify({"respuesta_havi": "¡Qué onda! ¿En qué te ayudo?"}), 200


@app.route('/api/chat-havi', methods=['POST'])
def chat_havi():
    try:
        data = request.json
        uid = str(data.get('id_cliente', ''))
        user_message = data.get('mensaje', '').strip()

        if not user_message:
            return jsonify({"status": "error", "message": "Mensaje vacío"}), 400

        logging.info(f"💬 [{uid}]: {user_message}")

        cliente = db_clientes.get(uid)
        
        if not cliente:
            ficha = "Cliente nuevo sin historial. Trátalo como prospecto."
            perfil_texto = "Usuario General"
        else:
            ficha = construir_ficha_inteligente(cliente)
            perfil_texto = cliente.get('perfil_negocio', 'N/D')

        system = construir_system_prompt(ficha)
        
        # --- ARMAR HISTORIAL CONVERSACIONAL ---
        historial = ""
        for turno in memoria[uid]:
            actor = "Usuario" if turno["rol"] == "user" else "HaviEr"
            historial += f"{actor}: {turno['msg']}\n"
        
        prompt_completo = f"{historial}Usuario: {user_message}\nHaviEr:"

        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt_completo,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.7,
            )
        )
        
        texto = response.text
        if not texto:
            texto = "Mmm, déjame procesar eso de otra forma. ¿Me lo repites?"
        
        # Guardar en memoria
        memoria[uid].append({"rol": "user", "msg": user_message})
        memoria[uid].append({"rol": "havi", "msg": texto})
        
        return jsonify({
            "status": "success",
            "perfil_detectado": perfil_texto,
            "respuesta_havi": texto
        })

    except Exception as e:
        logging.error(f"⚠️ Error chat: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Sistemas en mantenimiento."}), 500


@app.route('/api/reset-memoria', methods=['POST'])
def reset_memoria():
    """Limpia el historial de un usuario (útil para demos)."""
    uid = str(request.json.get('id_cliente', ''))
    if uid in memoria:
        del memoria[uid]
    return jsonify({"status": "success"})


if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug_mode, port=port, host='0.0.0.0')