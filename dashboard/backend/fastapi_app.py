# fastapi_app.py avec le framework FastAPI qui s'éxécute avec le serveur Uvicorn 

#1- sur le termile1:
# cd api puis   uvicorn fastapi_app:app --reload 

#fastapi_app c'est mon fichier sans l'extension .py
#app fait référence à l'instance de FastAPI que j'ai créée
#--reload permet au serveur de redémarrer automatiquement lorsque je modifie le code

#2 -puis  http://127.0.0.1:8000/clients

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import os

# Création de l'application FastAPI
app = FastAPI()

# Chemin du répertoire courant
current_directory = os.path.dirname(os.path.realpath(__file__))

# Charger le modèle LightGBM
model = joblib.load(os.path.join(current_directory, "model.joblib"))

# Charger les données des nouveaux clients
new_clients_df = pd.read_csv(os.path.join(current_directory, 'df_nouveaux_clients.csv'))

# Modèle de données entrantes
class PredictionRequest(BaseModel):
    SK_ID_CURR: int

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Prédire la probabilité pour un client spécifique et fournir des explications SHAP.
    """
    # Récupérer l'identifiant client
    sk_id_curr = request.SK_ID_CURR

    # Filtrer les données pour le client correspondant
    sample = new_clients_df[new_clients_df['SK_ID_CURR'] == sk_id_curr]

    if sample.empty:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    # Supprimer la colonne ID pour la prédiction
    sample = sample.drop(columns=['SK_ID_CURR'])

    # Prédire la probabilité
    prediction = model.predict_proba(sample)
    proba = prediction[0][1]  # Probabilité de la seconde classe

    # Calculer les valeurs SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Construire la réponse
    return {
        "probability": proba * 100,
        "shap_values": shap_values[1][0].tolist(),
        "feature_names": sample.columns.tolist(),
        "feature_values": sample.values[0].tolist()
    }

# Pour exécuter l'application :
# uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
