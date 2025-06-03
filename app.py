from flask import Flask, render_template, request, jsonify
import pandas as pd
from nyoka import PMML43 as pml

app = Flask(__name__)

# Cargar modelo
model = pml.parse("modelo.pmml")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)