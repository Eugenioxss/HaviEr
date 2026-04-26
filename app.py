import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# Cargar la API Key
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración de Gemini 2.5
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# --- MAGIA DE DATOS AQUÍ ---
# Cargamos el JSON a memoria una sola vez al prender el servidor
try:
    with open('clientes_etiquetados.json', 'r', encoding='utf-8') as f:
        clientes_data = json.load(f)
        
        # 1. Creamos el diccionario vacío
        db_clientes = {}
        
        # 2. Llenamos el diccionario usando 'user_id' (que es el nombre real en tu JSON)
        for c in clientes_data:
            # .get() evita que el código truene si falta una columna
            id_cliente = str(c.get('user_id', '')) 
            perfil = c.get('perfil_negocio', 'Perfil Desconocido')
            
            if id_cliente:
                db_clientes[id_cliente] = perfil

    print(f"✅ ¡Base de datos cargada! {len(db_clientes)} clientes etiquetados listos.")

except FileNotFoundError:
    print("⚠️ No se encontró el JSON. Havi funcionará con perfil general.")
    db_clientes = {}

@app.route('/', methods=['GET'])
def home():
    return "¡El cerebro de Havi está en línea y conectado a la red de PyTorch! 🧠🚀"

@app.route('/api/chat-havi', methods=['POST'])
def chat_havi():
    try:
        data = request.json
        # Ahora recibimos un ID, no el texto del perfil
        id_cliente = str(data.get('id_cliente', ''))
        user_message = data.get('mensaje', '')

        # Buscamos al cliente. Si pone un ID que no existe, le damos un perfil default.
        user_profile = db_clientes.get(id_cliente, 'Usuario General: Cliente estándar, tratar con amabilidad.')

        # Inyectamos tu hallazgo matemático directo a la IA
        contexto = (
            f"Eres Havi, la IA proactiva de Hey Banco. "
            f"El usuario con el que hablas tiene este diagnóstico detectado por nuestro modelo: {user_profile}. "
            "Tu tono es futurista, simple, eficiente y humano. "
            "Usa esta información para personalizar tu respuesta y retenerlo o ayudarlo sin ser invasivo. No menciones el nombre del clúster."
        )

        prompt_final = f"{contexto}\n\nUsuario dice: {user_message}\n\nHavi dice:"
        
        response = model.generate_content(prompt_final)
        
        return jsonify({
            "status": "success",
            "perfil_detectado": user_profile, # Se lo mandamos al frontend para que tú veas si le atinó
            "respuesta_havi": response.text
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)