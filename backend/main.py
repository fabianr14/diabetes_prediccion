from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nyoka import PMML43 as pml
import uvicorn

# Cargar modelo PMML
model = pml.parse("model.pmml")

app = FastAPI()

class DatosPaciente(BaseModel):
    genero: str
    edad: float
    hipertension: int
    enfermedad_cardiaca: int
    IMC: float
    nivel_de_HbA1c: float
    cantidad_de_glucosa: int
    diabetes: int

@app.post("/predecir")
async def predecir(datos: DatosPaciente):
    # Convertir datos a diccionario
    data_dict = datos.dict()
    
    # Aquí debes ajustar los nombres según el modelo PMML
    input_data = [[data_dict[key] for key in sorted(data_dict)]]
    
    try:
        resultado = model.predict(input_data)
        return {"prediccion": resultado[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
