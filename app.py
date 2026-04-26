import os
import json
import logging
import time
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from collections import defaultdict, deque
from datetime import datetime
from functools import wraps
from openai import OpenAI

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ FALTA GEMINI_API_KEY en el .env")

app = Flask(__name__)

# CORS restringido por variable de entorno
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
CORS(app, origins=ALLOWED_ORIGINS)

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

# MiMo Client Configuration
MIMO_API_KEY = os.getenv("MIMO_API_KEY")
mimo_client = None
if MIMO_API_KEY:
    mimo_client = OpenAI(
        api_key=MIMO_API_KEY,
        base_url="https://api.xiaomimimo.com/v1"
    )
    logging.info("✅ MiMo client initialized")
else:
    logging.warning("⚠️ MIMO_API_KEY not found in .env - MiMo fallback disabled")

# ============================================================
# ESTADO EN RAM
# ============================================================
memoria = defaultdict(lambda: deque(maxlen=6))
USUARIOS_NUEVOS = {}
INTERACCIONES_LOG = deque(maxlen=500)
RATE_LIMIT = defaultdict(list)
MAX_REQ_PER_MIN = 15

# ============================================================
# CARGA DE DATOS (con liberación de memoria)
# ============================================================
db_clientes = {}
try:
    with open('clientes_etiquetados.json', 'r', encoding='utf-8') as f:
        clientes_data = json.load(f)
        db_clientes = {
            str(c.get('user_id', '')): c
            for c in clientes_data if c.get('user_id')
        }
    del clientes_data
    gc.collect()
    logging.info(f"✅ Cerebro cargado: {len(db_clientes)} perfiles 360°")
except Exception as e:
    logging.error(f"❌ Error cargando JSON: {e}")

# ============================================================
# CONFIGURACIÓN DE CLUSTERS (DEC)
# ============================================================
CLUSTER_CONFIG = {
    0: {
        "nombre": "The Ghost",
        "tono": "Cercano, motivacional, paciente",
        "ofertas": ["Activación de cuenta", "Primer producto sin comisiones"],
        "trigger": "Inactividad prolongada"
    },
    1: {
        "nombre": "Mainstream Active",
        "tono": "Amigable, directo, práctico",
        "ofertas": ["Tarjeta de crédito", "Inversión básica", "Cashback"],
        "trigger": "Uso recurrente, oportunidad de upsell"
    },
    2: {
        "nombre": "Leveraged Ambitious",
        "tono": "Estratégico, educativo, advertencia financiera",
        "ofertas": ["Consolidación de deuda", "Asesoría financiera", "Refinanciamiento"],
        "trigger": "Alto endeudamiento, riesgo crediticio"
    },
    3: {
        "nombre": "Premium Healthy",
        "tono": "Sofisticado, exclusivo, consultor",
        "ofertas": ["Inversiones premium", "Tarjeta Black", "Patrimonio"],
        "trigger": "Alto poder adquisitivo, salud financiera"
    },
    4: {
        "nombre": "The Lukewarm",
        "tono": "Curioso, retador, gamificado",
        "ofertas": ["Retos de ahorro", "Promociones limitadas", "Productos novedosos"],
        "trigger": "Engagement medio, potencial de activación"
    }
}

# ============================================================
# UTILIDADES
# ============================================================
def rate_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        ahora = time.time()
        RATE_LIMIT[ip] = [t for t in RATE_LIMIT[ip] if ahora - t < 60]
        if len(RATE_LIMIT[ip]) >= MAX_REQ_PER_MIN:
            return jsonify({"error": "Demasiadas peticiones, espera un momento"}), 429
        RATE_LIMIT[ip].append(ahora)
        return f(*args, **kwargs)
    return wrapper


def log_interaccion(uid, tipo, mensaje_user="", respuesta_havi=""):
    INTERACCIONES_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "user_id": uid,
        "tipo": tipo,
        "mensaje_user": mensaje_user[:200],
        "respuesta_havi": respuesta_havi[:200],
        "es_nuevo": uid in USUARIOS_NUEVOS
    })


def clasificar_nuevo_usuario(data):
    """Heurística ponderada para asignar cluster a usuarios nuevos."""
    edad = data['edad']
    ingreso = data['ingreso_mensual']
    productos = data['num_productos']
    interes = data['interes_recomendaciones']

    if ingreso >= 50000 and productos >= 2:
        return 3  # Premium Healthy
    if ingreso >= 20000 and productos >= 3 and edad < 45:
        return 2  # Leveraged Ambitious
    if productos == 0 or interes <= 2:
        return 0  # The Ghost
    if interes >= 4 and productos >= 1:
        return 1  # Mainstream Active
    return 4  # The Lukewarm


def obtener_perfil(uid):
    """Retorna el perfil del usuario, ya sea de la DB o de los nuevos."""
    if uid in db_clientes:
        return db_clientes[uid]
    if uid in USUARIOS_NUEVOS:
        return USUARIOS_NUEVOS[uid]
    return None


def llamar_gemini(prompt, temperatura=0.7):
    """Wrapper unificado para llamadas a Gemini con manejo de errores."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperatura,
                max_output_tokens=400
            )
        )
        return response.text.strip(), None
    except genai_errors.ClientError as e:
        if "429" in str(e):
            logging.warning("⚠️ Cuota de Gemini agotada")
            return None, ("Estoy recibiendo muchas consultas. Intenta en un momento.", 429)
        logging.error(f"❌ Error Gemini: {e}")
        return None, ("Error temporal con el asistente.", 500)
    except Exception as e:
        logging.error(f"❌ Error inesperado: {e}")
        return None, ("Error interno del servidor.", 500)


def llamar_mimo(prompt, temperatura=0.7):
    """Wrapper para llamadas a MiMo con manejo de errores."""
    if not mimo_client:
        return None, ("Servicio de MiMo no disponible.", 503)
    
    try:
        response = mimo_client.chat.completions.create(
            model="mimo-v2.5-pro",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1024,
            temperature=temperatura,
            top_p=0.95,
            stream=False,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        logging.error(f"❌ Error MiMo: {e}")
        return None, ("Error temporal con el asistente.", 500)


def llamar_con_fallback(prompt, temperatura=0.7):
    """Wrapper con fallback configurable según FALLBACK_PRIORITY."""
    priority_str = os.getenv("FALLBACK_PRIORITY", "mimo,gemini")
    priorities = [p.strip().lower() for p in priority_str.split(",")]
    
    last_error = None
    
    for model in priorities:
        if model == "mimo":
            texto, error = llamar_mimo(prompt, temperatura)
            if error is None:
                return texto, None
            last_error = error
        elif model == "gemini":
            texto, error = llamar_gemini(prompt, temperatura)
            if error is None:
                return texto, None
            last_error = error
    
    return None, last_error


# ============================================================
# ENDPOINTS DE SALUD / KEEP-ALIVE
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "clientes_cargados": len(db_clientes),
        "usuarios_nuevos": len(USUARIOS_NUEVOS),
        "interacciones_log": len(INTERACCIONES_LOG)
    })


@app.route('/ping', methods=['GET'])
def ping():
    """Para cron-job.org - evita spin-down de Render."""
    return jsonify({"pong": True, "ts": time.time()})


# ============================================================
# SIGNUP
# ============================================================
@app.route('/api/signup', methods=['POST'])
@rate_limit
def signup():
    data = request.get_json() or {}
    requeridos = ['nombre', 'edad', 'ingreso_mensual', 'num_productos', 'interes_recomendaciones']
    faltantes = [c for c in requeridos if c not in data]
    if faltantes:
        return jsonify({"error": f"Faltan campos: {faltantes}"}), 400

    # Validación de tipos
    try:
        data['edad'] = int(data['edad'])
        data['ingreso_mensual'] = float(data['ingreso_mensual'])
        data['num_productos'] = int(data['num_productos'])
        data['interes_recomendaciones'] = int(data['interes_recomendaciones'])
        if not (18 <= data['edad'] <= 100):
            return jsonify({"error": "Edad fuera de rango"}), 400
        if data['ingreso_mensual'] < 0:
            return jsonify({"error": "Ingreso inválido"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Tipos de datos inválidos"}), 400

    cluster = clasificar_nuevo_usuario(data)
    nuevo_id = f"NEW_{int(time.time() * 1000)}"

    perfil = {
        "user_id": nuevo_id,
        "nombre": data['nombre'],
        "edad": data['edad'],
        "ingreso_mensual": data['ingreso_mensual'],
        "num_productos": data['num_productos'],
        "interes_recomendaciones": data['interes_recomendaciones'],
        "cluster": cluster,
        "cluster_nombre": CLUSTER_CONFIG[cluster]["nombre"],
        "fecha_registro": datetime.now().isoformat()
    }
    USUARIOS_NUEVOS[nuevo_id] = perfil

    # Bienvenida personalizada
    config = CLUSTER_CONFIG[cluster]
    prompt = f"""Eres HaviEr, asistente financiero de Hey Banco.
Da una bienvenida BREVE (máx 3 líneas) a {data['nombre']}.
Tono: {config['tono']}.
Menciona sutilmente una de estas oportunidades: {', '.join(config['ofertas'])}.
NO uses emojis excesivos. Sé natural y humano."""

    texto, error = llamar_con_fallback(prompt, temperatura=0.8)
    if error:
        bienvenida = f"¡Hola {data['nombre']}! Bienvenido a Hey Banco. Soy HaviEr, tu asistente. ¿En qué te ayudo?"
    else:
        bienvenida = texto

    log_interaccion(nuevo_id, "signup", "", bienvenida)

    return jsonify({
        "user_id": nuevo_id,
        "cluster": cluster,
        "cluster_nombre": config["nombre"],
        "bienvenida": bienvenida,
        "perfil": perfil
    })


# ============================================================
# CHAT
# ============================================================
@app.route('/api/chat-havi', methods=['POST'])
@rate_limit
def chat_havi():
    data = request.get_json() or {}
    uid = str(data.get('user_id', '')).strip().upper()
    user_message = data.get('mensaje', '').strip()

    if not uid or not user_message:
        return jsonify({"error": "Faltan user_id o mensaje"}), 400

    perfil = obtener_perfil(uid)
    if not perfil:
        return jsonify({"error": "Usuario no encontrado"}), 404

    # ✅ USAR CLUSTER_ID (no 'cluster')
    cluster = perfil.get('cluster_id', 4)
    config = CLUSTER_CONFIG.get(cluster, CLUSTER_CONFIG[4])

    # ✅ EXTRAER DATOS REALES
    edad = perfil.get('edad', 'N/D')
    sexo = perfil.get('sexo', '')
    ciudad = perfil.get('ciudad', 'México')
    ocupacion = perfil.get('ocupacion', 'N/D')
    ingreso = perfil.get('ingreso_mensual_mxn', 0)
    saldo = perfil.get('saldo_total_productos', 0)
    limite_credito = perfil.get('limite_credito_total', 0)
    utilizacion = perfil.get('utilizacion_promedio', 0)
    deuda_estimada = limite_credito * utilizacion
    score_buro = perfil.get('score_buro', 'N/D')
    es_hey_pro = perfil.get('es_hey_pro', False)
    tiene_seguro = perfil.get('tiene_seguro', False)
    productos_activos = perfil.get('num_productos_activos', 0)
    perfil_negocio = perfil.get('perfil_negocio', '')
    insight_trans = perfil.get('insight_transaccional', '')
    categoria_top = perfil.get('categoria_principal', 'N/D')
    cashback = perfil.get('cashback_total', 0)
    satisfaccion = perfil.get('satisfaccion_1_10', 'N/D')
    dias_inactivo = perfil.get('dias_desde_ultima_transaccion', 0)
    
    # Forma de tratamiento (sin nombre real)
    saludo_genero = "" 
    if sexo == "M": saludo_genero = "cliente"
    elif sexo == "H": saludo_genero = "cliente"
    else: saludo_genero = "cliente"

    # Historial
    historial_txt = ""
    for turno in list(memoria[uid])[-4:]:
        actor = "Usuario" if turno["rol"] == "user" else "HaviEr"
        historial_txt += f"{actor}: {turno['msg']}\n"

    prompt = f"""Eres HaviEr, asistente financiero virtual de Hey Banco. Hablas en español mexicano, eres breve (máx 3 líneas), cálido, claro y útil. NUNCA inventes datos.

═══ DATOS REALES DEL CLIENTE (úsalos textualmente, no los inventes) ═══
- ID: {uid}
- Edad: {edad} años | Ciudad: {ciudad} | Ocupación: {ocupacion}
- Ingreso mensual: ${ingreso:,.0f} MXN
- Saldo total en productos: ${saldo:,.2f} MXN
- Línea de crédito total: ${limite_credito:,.0f} MXN
- Utilización de crédito: {utilizacion*100:.1f}%
- Deuda estimada: ${deuda_estimada:,.0f} MXN
- Score de buró: {score_buro}
- Hey Pro: {"Sí" if es_hey_pro else "No"} | Seguro: {"Sí" if tiene_seguro else "No"}
- Productos activos: {productos_activos}
- Cashback acumulado: ${cashback:,.2f} MXN
- Categoría top de gasto: {categoria_top}
- Días desde última transacción: {dias_inactivo}
- Satisfacción autoreportada: {satisfaccion}/10

═══ SEGMENTACIÓN ═══
- Cluster: {config['nombre']}
- Perfil de negocio: {perfil_negocio}
- Insight transaccional: {insight_trans}
- Tono recomendado: {config['tono']}
- Productos relevantes para ofrecer: {', '.join(config['ofertas'])}

═══ HISTORIAL DE LA CONVERSACIÓN ═══
{historial_txt if historial_txt else "(Primera interacción del día)"}

═══ MENSAJE DEL USUARIO ═══
"{user_message}"

 ═══ REGLAS ESTRICTAS ═══
 1. NUNCA uses placeholders como [Cantidad], [Nombre], [Saldo]. Usa los números reales de arriba.
 2. Si no tienes el nombre, dirígete como "{saludo_genero}" o sin saludo si ya hubo conversación.
 3. Si el usuario pregunta por su saldo, dale el número real: ${saldo:,.2f}.
 4. Si pregunta por su deuda o crédito, usa los datos reales de arriba.
 5. NO repitas saludos si el historial ya muestra interacción.
 6. Si sugieres una oferta, hazlo natural y al final, no fuerces venta.
 7. Responde SOLO el mensaje al usuario. Nada de meta-comentarios ni etiquetas."""

    texto, error = llamar_con_fallback(prompt, temperatura=0.7)
    if error:
        return jsonify({"error": error[0]}), error[1]

    # Guardar memoria
    memoria[uid].append({"rol": "user", "msg": user_message})
    memoria[uid].append({"rol": "havi", "msg": texto})
    log_interaccion(uid, "chat", user_message, texto)

    return jsonify({
        "respuesta": texto,
        "cluster": cluster,
        "cluster_nombre": config["nombre"]
    })



# ============================================================
# INSIGHT PROACTIVO
# ============================================================
@app.route('/api/insight-proactivo', methods=['POST'])
@rate_limit
def insight_proactivo():
    data = request.get_json() or {}
    print("=" * 60)
    print(f"🟡 INSIGHT recibió: {data}")
    print("=" * 60)
    uid = str(data.get('user_id', ''))

    perfil = obtener_perfil(uid)
    if not perfil:
        return jsonify({"error": "Usuario no encontrado"}), 404

    cluster = perfil.get('cluster', 4)
    config = CLUSTER_CONFIG.get(cluster, CLUSTER_CONFIG[4])

    prompt = f"""Eres HaviEr, asistente financiero de Hey Banco.
Genera un INSIGHT PROACTIVO breve (máx 3 líneas) para este cliente:

- Nombre: {perfil.get('nombre', 'Cliente')}
- Segmento: {config['nombre']}
- Trigger: {config['trigger']}
- Tono: {config['tono']}

El insight debe ser una observación útil, una sugerencia o una alerta financiera relevante a su perfil. NO sea genérico."""

    texto, error = llamar_con_fallback(prompt, temperatura=0.8)
    if error:
        return jsonify({"error": error[0]}), error[1]

    log_interaccion(uid, "proactivo", "", texto)

    return jsonify({
        "insight": texto,
        "cluster": cluster,
        "cluster_nombre": config["nombre"]
    })


# ============================================================
# DASHBOARD
# ============================================================
@app.route('/api/dashboard/metricas', methods=['GET'])
def dashboard_metricas():
    """Métricas globales para el dashboard."""
    distribucion = defaultdict(int)
    for c in db_clientes.values():
        distribucion[c.get('cluster', -1)] += 1
    for c in USUARIOS_NUEVOS.values():
        distribucion[c.get('cluster', -1)] += 1

    return jsonify({
        "total_clientes": len(db_clientes) + len(USUARIOS_NUEVOS),
        "clientes_db": len(db_clientes),
        "usuarios_nuevos": len(USUARIOS_NUEVOS),
        "interacciones_totales": len(INTERACCIONES_LOG),
        "distribucion_clusters": {
            CLUSTER_CONFIG.get(k, {}).get("nombre", f"Cluster {k}"): v
            for k, v in distribucion.items()
        },
        "memoria_activa": len(memoria)
    })


@app.route('/api/dashboard/cliente/<uid>', methods=['GET'])
def dashboard_cliente(uid):
    """Perfil completo + historial de chat de un cliente."""
    perfil = obtener_perfil(uid)
    if not perfil:
        return jsonify({"error": "Usuario no encontrado"}), 404

    cluster = perfil.get('cluster', 4)
    config = CLUSTER_CONFIG.get(cluster, CLUSTER_CONFIG[4])
    historial = list(memoria.get(uid, []))

    return jsonify({
        "perfil": perfil,
        "cluster_info": config,
        "historial_chat": historial,
        "total_turnos": len(historial)
    })


@app.route('/api/dashboard/feed', methods=['GET'])
def dashboard_feed():
    """Feed en tiempo real de las últimas interacciones."""
    return jsonify({
        "total": len(INTERACCIONES_LOG),
        "interacciones": list(INTERACCIONES_LOG)[-50:][::-1]
    })


@app.route('/api/clientes', methods=['GET'])
def listar_clientes():
    """Listado paginado de clientes."""
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 50)), 200)
    cluster_filter = request.args.get('cluster')

    todos = list(db_clientes.values()) + list(USUARIOS_NUEVOS.values())
    if cluster_filter is not None:
        try:
            cf = int(cluster_filter)
            todos = [c for c in todos if c.get('cluster') == cf]
        except ValueError:
            pass

    start = (page - 1) * per_page
    end = start + per_page

    return jsonify({
        "total": len(todos),
        "page": page,
        "per_page": per_page,
        "clientes": todos[start:end]
    })


@app.route('/api/reset-memoria', methods=['POST'])
def reset_memoria():
    data = request.get_json() or {}
    uid = str(data.get('user_id', ''))
    if uid in memoria:
        del memoria[uid]
        return jsonify({"status": "ok", "mensaje": f"Memoria de {uid} reseteada"})
    return jsonify({"status": "ok", "mensaje": "No había memoria activa"})


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

@app.route("/api/cliente/<user_id>", methods=["GET"])
def get_cliente(user_id):
    cliente = next((c for c in clientes_data if c["user_id"] == user_id), None)
    if not cliente:
        return jsonify({"error": "Cliente no encontrado"}), 404
    return jsonify({
        "user_id": cliente["user_id"],
        "cluster": cliente.get("cluster_name", "N/A"),
        "edad": cliente.get("edad", "N/A")
    })
