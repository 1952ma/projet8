
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
from fastapi.responses import JSONResponse
import shap

# Charger le modèle LightGBM
current_directory = os.path.dirname(os.path.realpath(__file__))
model = joblib.load(os.path.join(current_directory, "model.joblib"))

# Création de l'application FastAPI
app = FastAPI()

# La structure de la requête de prédiction
class ClientData(BaseModel):
    SK_ID_CURR: int

# Chargement des données des nouveaux clients (CSV)
new_clients_df = pd.read_csv(os.path.join(current_directory, 'df_nouveaux_clients.csv'))

# Route de redirection de '/' vers '/clients'
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'application de prédiction de crédit."}

# Route pour la liste des clients : http://127.0.0.1:8000/clients
@app.get("/clients")
def get_clients():
    """Retourner la liste des SK_ID_CURR"""
    return new_clients_df['SK_ID_CURR'].tolist()

# Route pour prédire un client spécifique : http://127.0.0.1:8000/predict
@app.post("/predict")
def predict(client_data: ClientData):
    """Faire une prédiction pour un client spécifique"""
    try:
        # Récupérer le SK_ID_CURR
        client_id = client_data.SK_ID_CURR

        # Récupérer les données du client
        client_row = new_clients_df[new_clients_df['SK_ID_CURR'] == client_id]

        # Vérification si les données du client existent
        if client_row.empty:
            raise HTTPException(status_code=404, detail="Client not found")

        # Préparer les données pour le modèle (supprimer SK_ID_CURR)
        client_features = client_row.drop(columns=["SK_ID_CURR"]).values
        
        # Effectuer la prédiction
        prediction_proba = float(model.predict_proba(client_features)[:, 1][0])
        prediction_label = int(prediction_proba > 0.53)

        # Calculer l'importance des features pour ce client
        feature_importances = model.feature_importances_

        return {
            "SK_ID_CURR": int(client_id),
            "probability": prediction_proba,
            "prediction_label": prediction_label,
            "feature_importances": feature_importances.tolist()  # Convertir en liste pour JSON
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route pour obtenir le graphique SHAP : http://127.0.0.1:8000/shap
import base64
import io
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse

# Route pour obtenir le graphique SHAP : http://127.0.0.1:8000/shap
@app.post("/shap")
def generate_shap(client_data: ClientData):
    """Génère un graphique SHAP pour un client spécifique"""
    try:
        # Récupérer le SK_ID_CURR
        client_id = client_data.SK_ID_CURR

        # Récupérer les données du client
        client_row = new_clients_df[new_clients_df['SK_ID_CURR'] == client_id]

        # Vérification si les données du client existent
        if client_row.empty:
            raise HTTPException(status_code=404, detail="Client not found")

        # Préparer les données pour SHAP (supprimer SK_ID_CURR)
        client_features = client_row.drop(columns=["SK_ID_CURR"]).values

        # Créer un explainer SHAP pour LightGBM
        explainer = shap.Explainer(model, new_clients_df.drop(columns=["SK_ID_CURR"]))
        shap_values = explainer(client_features)

        # Générer le graphique SHAP
        plt.figure()
        shap.waterfall_plot(shap_values[0], max_display=10)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()

        # Convertir en base64
        base64_image = base64.b64encode(buf.read()).decode('utf-8')

        return {"shap_graph": base64_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



















