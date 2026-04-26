import os, json, logging, time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors  # 🆕 Para manejar quota
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

# 🆕 USUARIOS CREADOS VÍA /api/signup (no están en el JSON)
USUARIOS_NUEVOS = {}

# --- CARGA DE DATOS ---
db_clientes = {}
try:
    with open('clientes_etiquetados.json', 'r', encoding='utf-8') as f:
        clientes_data = json.load(f)
        db_clientes = {str(c.get('user_id', '')): c for c in clientes_data if c.get('user_id')}
    logging.info(f"✅ Cerebro cargado: {len(db_clientes)} perfiles 360°")
except Exception as e:
    logging.error(f"❌ Error cargando JSON: {e}")


# ============================================================
# 🆕 ESTRATEGIAS POR CLUSTER DEC
# ============================================================
CLUSTER_DEC_STRATEGIES = {
    0: {
        "nombre": "El Fantasma",
        "tono": "directo, urgente, con incentivo fuerte",
        "ofrecer": ["bono de reactivación $300", "tarjeta sin anualidad", "cashback inmediato"],
        "evitar": ["productos complejos", "inversiones", "lenguaje formal"],
        "trigger": "necesita un motivo poderoso para volver a usar Hey"
    },
    1: {
        "nombre": "Mainstream Activo",
        "tono": "cercano, fresco, motivacional",
        "ofrecer": ["cashback en categorías favoritas", "metas de ahorro", "tarjetas premium"],
        "evitar": ["asustarlo con riesgos", "ofertas de crédito agresivas"],
        "trigger": "engagement diario, gamificación de hábitos"
    },
    2: {
        "nombre": "Apalancado Ambicioso",
        "tono": "estratégico, retador, mentor financiero",
        "ofrecer": ["optimización de score", "co-piloto de crédito", "consolidación de deudas"],
        "evitar": ["sermonearlo", "tono paternalista"],
        "trigger": "quiere crecer su capacidad financiera, no que le digan que no"
    },
    3: {
        "nombre": "Premium Saludable",
        "tono": "sofisticado, asesor wealth, exclusivo",
        "ofrecer": ["inversiones", "seguros patrimoniales", "VIP status", "asesoría premium"],
        "evitar": ["promociones masivas", "lenguaje básico"],
        "trigger": "busca optimizar patrimonio y reconocimiento"
    },
    4: {
        "nombre": "Tibio",
        "tono": "empático, indagador, generador de conexión",
        "ofrecer": ["personalización", "encuestas cortas", "primer producto adicional"],
        "evitar": ["bombardearlo con productos", "spam"],
        "trigger": "está a punto de churnear, hay que entender por qué"
    }
}


# ============================================================
# 🧠 PRE-PROCESADOR DE INSIGHTS (cliente existente)
# ============================================================
def construir_ficha_inteligente(c: dict) -> str:
    """Convierte el JSON crudo en un brief accionable para la IA."""
    
    alertas = []
    oportunidades = []
    
    if c.get('score_buro', 850) < 600:
        alertas.append(f"🚨 Score buró BAJO: {c.get('score_buro')} (riesgo impago)")
    
    util = c.get('utilizacion_promedio', 0)
    if util > 0.7:
        alertas.append(f"🚨 Usa {util*100:.0f}% de su crédito (saturado)")
    elif util > 0.3:
        alertas.append(f"⚠️ Usa {util*100:.0f}% de su crédito")
    
    dias_inactivo = c.get('dias_desde_ultima_transaccion', 0)
    if dias_inactivo > 15:
        alertas.append(f"⚠️ {dias_inactivo} días sin transaccionar")
    
    var = c.get('variacion_actividad', 0)
    if var < -0.3:
        alertas.append(f"📉 Actividad cayó {abs(var)*100:.0f}% vs mes pasado")
    
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


# ============================================================
# 🆕 FICHA PARA USUARIO NUEVO (sin historial transaccional)
# ============================================================
def construir_ficha_usuario_nuevo(perfil: dict) -> str:
    return f"""
PERFIL DEC ASIGNADO POR SIMILITUD: {perfil['cluster_nombre']}
👤 NUEVO USUARIO (Día 1, sin historial transaccional aún)

DEMOGRAFÍA: {perfil['edad']} años, {perfil['ciudad']}, {perfil['estado']}
INGRESO DECLARADO: ${perfil['ingreso_mensual']:,} MXN/mes
PATRIMONIO ESTIMADO: ${int(perfil['patrimonio_estimado']):,} MXN

INTENCIÓN:
- Producto que más le interesa: {perfil['producto_interes']}
- Frecuencia transaccional esperada: {perfil['frecuencia_tx']}
- Productos en otros bancos: {perfil['num_productos']}
- Apertura a recomendaciones: {perfil['interes_recomendaciones']}/10

⚠️ NO asumas historial. Aprovecha para construir confianza y entender sus metas.
""".strip()


# ============================================================
# ✏️ SYSTEM INSTRUCTION (ahora con estrategia de cluster)
# ============================================================
def construir_system_prompt(ficha: str, cluster_id: int = None) -> str:
    """Si se pasa cluster_id, inyecta la estrategia DEC."""
    
    estrategia_extra = ""
    if cluster_id is not None and cluster_id in CLUSTER_DEC_STRATEGIES:
        e = CLUSTER_DEC_STRATEGIES[cluster_id]
        estrategia_extra = f"""

🎯 ESTRATEGIA DE SEGMENTO ({e['nombre']}):
- Tono recomendado: {e['tono']}
- Productos a priorizar: {', '.join(e['ofrecer'])}
- EVITAR: {', '.join(e['evitar'])}
- Insight clave: {e['trigger']}
"""
    
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
{estrategia_extra}
"""


# ============================================================
# 🆕 CLASIFICADOR DE SIGNUP (heurística ponderada)
# ============================================================
def clasificar_cluster_signup(datos: dict) -> dict:
    """Predice cluster DEC basado en datos de registro."""
    edad = datos.get('edad', 30)
    ingreso = datos.get('ingreso_mensual', 15000)
    num_productos = datos.get('num_productos', 1)
    interes = datos.get('interes_recomendaciones', 5)
    producto = datos.get('producto_interes', 'cuenta').lower()
    freq = datos.get('frecuencia_tx', 'media').lower()
    
    multiplicador = max(2, min(10, (edad - 18) * 0.3))
    patrimonio = ingreso * multiplicador
    
    s = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Cluster 3: Premium
    if ingreso >= 50000: s[3] += 3
    if patrimonio >= 200000: s[3] += 2
    if producto in ['inversion', 'seguro']: s[3] += 2
    if edad >= 35: s[3] += 1
    if num_productos >= 3: s[3] += 1
    
    # Cluster 2: Apalancado
    if producto == 'credito': s[2] += 3
    if 25 <= edad <= 40: s[2] += 2
    if 20000 <= ingreso <= 60000: s[2] += 2
    if freq == 'alta': s[2] += 1
    if num_productos >= 2: s[2] += 1
    
    # Cluster 1: Mainstream
    if freq == 'alta': s[1] += 3
    if producto in ['tarjeta', 'cuenta']: s[1] += 2
    if 18 <= edad <= 35: s[1] += 2
    if 10000 <= ingreso <= 35000: s[1] += 2
    if interes >= 7: s[1] += 1
    
    # Cluster 4: Tibio
    if freq == 'baja': s[4] += 2
    if interes <= 4: s[4] += 2
    if num_productos == 1: s[4] += 1
    
    # Cluster 0: Fantasma
    if freq == 'baja' and interes <= 3: s[0] += 3
    if ingreso < 10000: s[0] += 1
    
    pred = max(s, key=s.get)
    total = sum(s.values()) or 1
    
    return {
        "cluster_dec": pred,
        "confianza": round(s[pred] / total, 2),
        "scores": s,
        "patrimonio_estimado": patrimonio
    }


# ============================================================
# 🆕 ENCUENTRA CLIENTE "HERMANO" PARA NARRATIVA
# ============================================================
def encontrar_cliente_hermano(cluster_id: int, datos: dict) -> dict:
    """Busca el cliente más parecido del JSON dentro del mismo cluster."""
    candidatos = [c for c in db_clientes.values() 
                  if c.get('cluster_dec') == cluster_id or c.get('perfil_negocio_id') == cluster_id]
    if not candidatos:
        return None
    
    edad_s = datos.get('edad', 30)
    ingreso_s = datos.get('ingreso_mensual', 15000)
    
    def dist(c):
        d_edad = abs(c.get('edad', 30) - edad_s) / 50
        d_ing = abs(c.get('ingreso_mensual_mxn', 15000) - ingreso_s) / 100000
        return d_edad + d_ing
    
    return min(candidatos, key=dist)


# ============================================================
# 🆕 GENERADOR DE BIENVENIDA PERSONALIZADA
# ============================================================
def generar_bienvenida(perfil: dict, estrategia: dict, hermano: dict) -> str:
    contexto_hermano = ""
    if hermano:
        contexto_hermano = f"\nClientes similares del segmento '{estrategia['nombre']}' suelen enfocarse en: {', '.join(estrategia['ofrecer'][:2])}."
    
    prompt = f"""Eres HaviEr, asistente de Hey Banco. Un nuevo usuario acaba de registrarse.

DATOS:
- Edad: {perfil['edad']}, Ingreso: ${perfil['ingreso_mensual']:,}
- Ciudad: {perfil['ciudad']}, {perfil['estado']}
- Producto que le interesa: {perfil['producto_interes']}
- Segmento detectado: {estrategia['nombre']}
- Tono: {estrategia['tono']}
{contexto_hermano}

INSTRUCCIONES:
1. Saluda con calidez (sin presentarte formalmente).
2. Demuestra que ya entendiste su perfil sin sonar invasivo.
3. Sugiere 1 acción concreta basada en su producto de interés.
4. Termina con pregunta abierta.
5. Máx 80 palabras. Tono regio mexicano.
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt
        )
        return response.text.strip() if response.text else f"¡Qué onda! Veo que te interesa {perfil['producto_interes']}. ¿Le entramos?"
    except Exception as e:
        logging.error(f"Error bienvenida: {e}")
        return f"¡Qué onda! Veo que te interesa {perfil['producto_interes']}. Tengo ideas pa' ti — ¿le entramos?"


# ============================================================
# ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "clientes_cargados": len(db_clientes),
        "usuarios_nuevos": len(USUARIOS_NUEVOS)
    })


@app.route('/api/clientes', methods=['GET'])
def listar_clientes():
    return jsonify({
        "total": len(db_clientes),
        "ids": list(db_clientes.keys())[:50]
    })


# 🆕 SIGNUP
@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        required = ['edad', 'ingreso_mensual', 'estado', 'ciudad',
                    'num_productos', 'interes_recomendaciones',
                    'producto_interes', 'frecuencia_tx']
        for f in required:
            if f not in data:
                return jsonify({"error": f"Falta campo: {f}"}), 400
        
        clasif = clasificar_cluster_signup(data)
        cluster_id = clasif['cluster_dec']
        hermano = encontrar_cliente_hermano(cluster_id, data)
        estrategia = CLUSTER_DEC_STRATEGIES[cluster_id]
        
        nuevo_id = f"new_{int(time.time())}_{data['edad']}"
        
        perfil = {
            "user_id": nuevo_id,
            "es_nuevo": True,
            "edad": data['edad'],
            "ingreso_mensual": data['ingreso_mensual'],
            "estado": data['estado'],
            "ciudad": data['ciudad'],
            "num_productos": data['num_productos'],
            "interes_recomendaciones": data['interes_recomendaciones'],
            "producto_interes": data['producto_interes'],
            "frecuencia_tx": data['frecuencia_tx'],
            "cluster_dec": cluster_id,
            "cluster_nombre": estrategia['nombre'],
            "patrimonio_estimado": clasif['patrimonio_estimado'],
            "hermano_id": hermano.get('user_id') if hermano else None,
            "fecha_signup": datetime.now().isoformat()
        }
        
        USUARIOS_NUEVOS[nuevo_id] = perfil
        bienvenida = generar_bienvenida(perfil, estrategia, hermano)
        
        # Guardar en memoria conversacional
        memoria[nuevo_id].append({"rol": "havi", "msg": bienvenida})
        
        logging.info(f"✅ Signup: {nuevo_id} → Cluster {cluster_id} ({estrategia['nombre']})")
        
        return jsonify({
            "status": "success",
            "user_id": nuevo_id,
            "cluster_asignado": {
                "id": cluster_id,
                "nombre": estrategia['nombre'],
                "confianza": clasif['confianza']
            },
            "mensaje_bienvenida": bienvenida
        }), 200
    
    except genai_errors.ClientError as e:
        logging.error(f"⚠️ Quota Gemini: {e}")
        return jsonify({"error": "Servicio AI saturado, intenta en un momento"}), 429
    except Exception as e:
        logging.error(f"⚠️ Error signup: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/insight-proactivo', methods=['POST'])
def insight_proactivo():
    try:
        uid = str(request.json.get('id_cliente', ''))
        
        # ✏️ Detectar usuario nuevo primero
        if uid in USUARIOS_NUEVOS:
            perfil = USUARIOS_NUEVOS[uid]
            ficha = construir_ficha_usuario_nuevo(perfil)
            cluster_id = perfil['cluster_dec']
        else:
            cliente = db_clientes.get(uid)
            if not cliente:
                return jsonify({"respuesta_havi": "¡Qué onda! Soy HaviEr. ¿En qué te ayudo hoy?"})
            ficha = construir_ficha_inteligente(cliente)
            cluster_id = cliente.get('cluster_dec') or cliente.get('perfil_negocio_id')
        
        system = construir_system_prompt(ficha, cluster_id)
        
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

        # ✏️ Detectar tipo de usuario
        if uid in USUARIOS_NUEVOS:
            perfil = USUARIOS_NUEVOS[uid]
            ficha = construir_ficha_usuario_nuevo(perfil)
            cluster_id = perfil['cluster_dec']
            perfil_texto = f"Nuevo Usuario - {perfil['cluster_nombre']}"
        else:
            cliente = db_clientes.get(uid)
            if not cliente:
                ficha = "Cliente nuevo sin historial. Trátalo como prospecto."
                cluster_id = None
                perfil_texto = "Usuario General"
            else:
                ficha = construir_ficha_inteligente(cliente)
                cluster_id = cliente.get('cluster_dec') or cliente.get('perfil_negocio_id')
                perfil_texto = cliente.get('perfil_negocio', 'N/D')

        system = construir_system_prompt(ficha, cluster_id)
        
        # Historial
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
        
        texto = response.text or "Mmm, déjame procesar eso de otra forma. ¿Me lo repites?"
        
        memoria[uid].append({"rol": "user", "msg": user_message})
        memoria[uid].append({"rol": "havi", "msg": texto})
        
        return jsonify({
            "status": "success",
            "perfil_detectado": perfil_texto,
            "respuesta_havi": texto
        })

    except genai_errors.ClientError as e:
        logging.error(f"⚠️ Quota: {e}")
        return jsonify({"status": "error", "message": "Servicio saturado, intenta en un momento."}), 429
    except Exception as e:
        logging.error(f"⚠️ Error chat: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Sistemas en mantenimiento."}), 500


@app.route('/api/reset-memoria', methods=['POST'])
def reset_memoria():
    uid = str(request.json.get('id_cliente', ''))
    if uid in memoria:
        del memoria[uid]
    return jsonify({"status": "success"})


if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug_mode, port=port, host='0.0.0.0')

@app.route('/api/dashboard/metricas', methods=['GET'])
def dashboard_metricas():
    """Métricas globales para el dashboard."""
    total_clientes = len(db_clientes)
    total_nuevos = len(USUARIOS_NUEVOS)
    
    # Distribución por cluster
    dist_clusters = defaultdict(int)
    for c in db_clientes.values():
        cid = c.get('cluster_dec') or c.get('perfil_negocio_id', -1)
        dist_clusters[cid] += 1
    
    # Conversaciones activas
    conversaciones_activas = sum(1 for m in memoria.values() if len(m) > 0)
    total_interacciones = sum(len(m) for m in memoria.values())
    
    return jsonify({
        "total_clientes": total_clientes,
        "usuarios_nuevos_signup": total_nuevos,
        "conversaciones_activas": conversaciones_activas,
        "total_interacciones": total_interacciones,
        "distribucion_clusters": {
            CLUSTER_DEC_STRATEGIES[k]['nombre']: v 
            for k, v in dist_clusters.items() if k in CLUSTER_DEC_STRATEGIES
        }
    })


@app.route('/api/dashboard/cliente/<uid>', methods=['GET'])
def dashboard_cliente(uid):
    """Detalle completo de un cliente para el explorer."""
    uid = str(uid)
    if uid in USUARIOS_NUEVOS:
        return jsonify({"tipo": "nuevo", "data": USUARIOS_NUEVOS[uid]})
    cliente = db_clientes.get(uid)
    if not cliente:
        return jsonify({"error": "Cliente no encontrado"}), 404
    return jsonify({
        "tipo": "existente",
        "data": cliente,
        "historial_chat": list(memoria.get(uid, []))
    })


@app.route('/api/dashboard/conversaciones', methods=['GET'])
def dashboard_conversaciones():
    """Lista de conversaciones recientes."""
    convos = []
    for uid, turnos in memoria.items():
        if turnos:
            convos.append({
                "user_id": uid,
                "num_mensajes": len(turnos),
                "ultimo_mensaje": turnos[-1]['msg'][:100],
                "es_nuevo": uid in USUARIOS_NUEVOS
            })
    return jsonify({"conversaciones": convos[:50]})
