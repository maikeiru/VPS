from flask import Flask, request, jsonify, send_from_directory
from main import procesar_mensaje

app = Flask(__name__)

@app.route("/api/dodonest", methods=["POST", "OPTIONS"])
def api_dodonest():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = request.get_json()
    mensaje = data.get("mensaje", "")
    respuesta = procesar_mensaje(mensaje)
    return jsonify({"respuesta": respuesta})

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

