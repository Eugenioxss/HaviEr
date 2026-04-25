import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# Cargar la API Key desde el archivo .env
load_dotenv()

app = Flask(__name__)
CORS(app) # Importante para que tu Frontend no tenga errores de conexión

# Configuración de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-lite-001')

@app.route('/api/chat-havi', methods=['POST'])
def chat_havi():
    try:
        data = request.json
        # Recibimos el perfil que vendrá del análisis de ML (motor-ia)
        user_profile = data.get('perfil', 'Usuario General')
        user_message = data.get('mensaje', '')

        # El "System Prompt" define la personalidad que pide Hey Banco
        contexto = (
            f"Eres Havi, la IA proactiva de Hey Banco. "
            f"El usuario actual pertenece al segmento: {user_profile}. "
            "Tu tono es futurista, simple, eficiente y muy humano[cite: 446, 447]. "
            "Tu objetivo es transformar la banca ofreciendo soluciones antes de que el usuario las pida[cite: 437, 474]."
        )

        prompt_final = f"{contexto}\n\nUsuario dice: {user_message}\n\nHavi dice:"
        
        response = model.generate_content(prompt_final)
        
        return jsonify({
            "status": "success",
            "respuesta_havi": response.text
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # El puerto 5000 es el estándar, pero ngrok lo leerá fácil
    app.run(debug=True, port=5000)