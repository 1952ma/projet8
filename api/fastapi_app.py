
# fastapi_app.py avec le framework FastAPI qui s'éxécute avec le serveur Uvicorn 

#1- sur le termile1:
# cd api puis   uvicorn fastapi_app:app --reload 

#fastapi_app c'est mon fichier sans l'extension .py
#app fait référence à l'instance de FastAPI que j'ai créée
#--reload permet au serveur de redémarrer automatiquement lorsque je modifie le code

#2 -puis  http://127.0.0.1:8000/clients

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.responses import JSONResponse
import shap

current_directory = os.path.dirname(os.path.realpath(__file__))
model = joblib.load(os.path.join(current_directory, "model.joblib"))
app = FastAPI()

class ClientData(BaseModel):
    SK_ID_CURR: int

new_clients_df = pd.read_csv(os.path.join(current_directory, 'df_nouveaux_clients.csv'))
# Initialiser l'explainer SHAP avec LightGBM
explainer = shap.Explainer(model, new_clients_df)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'application de prédiction de crédit."}

@app.get("/clients")
def get_clients():
    return new_clients_df['SK_ID_CURR'].tolist()

@app.post("/predict")
def predict(client_data: ClientData):
    try:
        client_id = client_data.SK_ID_CURR
        client_row = new_clients_df[new_clients_df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            raise HTTPException(status_code=404, detail="Client not found")
        client_features = client_row.drop(columns=["SK_ID_CURR"]).values
        prediction_proba = float(model.predict_proba(client_features)[:, 1][0])
        prediction_label = int(prediction_proba > 0.53)

        feature_importances = model.feature_importances_
        return {
            "SK_ID_CURR": int(client_id),
            "probability": prediction_proba,
            "prediction_label": prediction_label,
            "feature_importances": feature_importances.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shap/{client_id}")
def generate_shap(client_id: int):
    try:
        client_row = new_clients_df[new_clients_df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            raise HTTPException(status_code=404, detail="Client introuvable")
        
        shap_values = explainer(client_row)

        # Graphique Waterfall Plot
        fig_waterfall = plt.figure()
        shap.waterfall_plot(shap_values[0], show=False)
        buf_waterfall = io.BytesIO()
        plt.savefig(buf_waterfall, format="png")
        buf_waterfall.seek(0)
        waterfall_encoded = base64.b64encode(buf_waterfall.getvalue()).decode()
        
        return JSONResponse(content={
            "shap_waterfall": waterfall_encoded
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
























