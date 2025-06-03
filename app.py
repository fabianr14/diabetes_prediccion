from flask import Flask, request, jsonify, render_template
from pypmml import Model

app = Flask(__name__, template_folder='templates')

# Cargar el modelo PMML
try:
    model = Model.fromFile('model.pmml')
except Exception as e:
    print("Error al cargar el modelo:", e)
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prediction = model.predict(data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
