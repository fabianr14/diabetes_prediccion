from flask import Flask, request, jsonify
from pypmml import Model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # permite conexiones desde tu frontend

# Cargar modelo PMML al inicio
model = Model.load('diabetes_model.pmml')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Asegurar que todos los campos estén presentes
        expected_fields = ["genero", "edad", "hipertension", "enfermedad cardiaca", "tabaquismo", "IMC", "nivel de HbA1c", "cantidad de glucosa"]
        if not all(field in data for field in expected_fields):
            return jsonify({"error": "Faltan campos en la entrada"}), 400

        # Predecir usando el modelo PMML
        prediction = model.predict(data)
        diabetes_risk = prediction.get("diabetes", 0)

        return jsonify({
            "prediccion": int(diabetes_risk),
            "mensaje": "Riesgo ALTO" if diabetes_risk == 1 else "Riesgo BAJO"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
