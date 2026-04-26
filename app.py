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

    texto, error = llamar_gemini(prompt, temperatura=0.8)
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
    print("=" * 50)
    print("📥 PAYLOAD RECIBIDO:", data)
    print("=" * 50)
    uid = str(data.get('user_id', ''))
    user_message = data.get('mensaje', '').strip()

    if not uid or not user_message:
        return jsonify({"error": "Faltan user_id o mensaje"}), 400

    perfil = obtener_perfil(uid)
    if not perfil:
        return jsonify({"error": "Usuario no encontrado"}), 404

    cluster = perfil.get('cluster', 4)
    config = CLUSTER_CONFIG.get(cluster, CLUSTER_CONFIG[4])

    # Historial truncado a últimos 4 turnos
    historial = ""
    for turno in list(memoria[uid])[-4:]:
        actor = "Usuario" if turno["rol"] == "user" else "HaviEr"
        historial += f"{actor}: {turno['msg']}\n"

    prompt = f"""Eres HaviEr, asistente financiero de Hey Banco.

PERFIL DEL CLIENTE:
- Nombre: {perfil.get('nombre', 'Cliente')}
- Segmento: {config['nombre']}
- Tono recomendado: {config['tono']}
- Ofertas relevantes: {', '.join(config['ofertas'])}

HISTORIAL RECIENTE:
{historial}

MENSAJE ACTUAL:
Usuario: {user_message}

Responde de forma BREVE (máx 4 líneas), natural y útil. Si es relevante, sugiere sutilmente una oferta. NO repitas saludos si ya hubo conversación previa."""

    texto, error = llamar_gemini(prompt, temperatura=0.7)
    if error:
        return jsonify({"error": error[0]}), error[1]

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

    texto, error = llamar_gemini(prompt, temperatura=0.8)
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
