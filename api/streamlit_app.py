# cd api

#sur le terminale 2: streamlit run streamlit_app.py
#si ConnectionError alors Streamlit ne parvient pas à se connecter à API FastAPI => FastAPI n'est pas en cours d'exécution sur le terminale1, il faut l'exécuter

import streamlit as st
import requests
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64  # Importation nécessaire pour décoder l'image
from io import BytesIO 
from PIL import Image
# Chargement du modèle et des données locales
def load_model_and_data():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model = joblib.load(os.path.join(current_directory, "model.joblib"))
    new_clients_df = pd.read_csv(os.path.join(current_directory, 'df_nouveaux_clients.csv'))
    description_feature_df = pd.read_csv(os.path.join(current_directory, 'description_feature.csv'))
    description_feature_df.columns = ['Feature', 'Description']
    return model, new_clients_df, description_feature_df
def get_client_ids(api_url):
    try:
        response = requests.get(f"{api_url}/clients")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des clients: {e}")
        st.stop()
def get_prediction(api_url, client_id):
    try:
        response = requests.post(f"{api_url}/predict", json={"SK_ID_CURR": client_id})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        return None
# Fonction pour obtenir les graphiques SHAP
def get_shap_graph(client_id):
    response = requests.post(f"{API_URL}/shap/{client_id}")
    response.raise_for_status()
    shap_data = response.json()
    return shap_data["shap_waterfall"]

# Définir les variables principales
API_URL = "http://127.0.0.1:8000"
model, new_clients_df, description_feature_df = load_model_and_data()
# --- Titre de l'application ---
st.title("Prédiction du score de crédit")
st.markdown(
    "Cette application permet d'évaluer le risque de crédit des clients. "
    "Elle inclut des outils interactifs d'analyse, de visualisation et d'explicabilité."
)

# --- Jauge de référence ---
st.header("Jauge de référence globale")
threshold = 0.53
st.markdown(
    f"<div style='position: relative; width: 100%; height: 30px; background: linear-gradient(to right, green {threshold*100}%, red {threshold*100}%);'></div>",
    unsafe_allow_html=True
)

# --- Sélection de l'ID client ---
st.subheader("Sélectionnez un ID client")
client_ids = get_client_ids(API_URL)
selected_client_id = st.selectbox("Liste des clients :", client_ids)

# --- Prédiction pour le client sélectionné ---
prediction_data = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if st.button("Réaliser une prédiction"):
    st.session_state.prediction_data = get_prediction(API_URL, selected_client_id)



# Affichage permanent de la jauge client
if st.session_state.prediction_data:
    probability = st.session_state.prediction_data["probability"]
    score_color = 'green' if probability < threshold else 'red'
    
    st.subheader(f"Résultat pour le client : {selected_client_id}")
    st.markdown(
        f"<div style='position: relative; width: {probability*100}%; height: 30px; background-color: {score_color}; text-align: center; color: white;'>"
        f"{'<b>Faible</b> risque' if score_color == 'green' else '<b>Haut</b> risque'} : {probability:.2%}</div>",
        unsafe_allow_html=True
    )

# --- Menu latéral ---
menu = st.sidebar.radio(
    "Menu",
    ['Selectionner :','Feature Importance Locale', 'Visualisation Features', 'Description Features', 'Feature Importance Globale']
)
if menu == 'Selectionner :':
    st.header("")
# --- Feature Importance Locale ---
elif menu == 'Feature Importance Locale':
    st.header("Feature Importance Locale")
    
    shap_waterfall = get_shap_graph(selected_client_id)

    # Afficher le Waterfall Plot
    waterfall_image = Image.open(BytesIO(base64.b64decode(shap_waterfall)))
    st.image(waterfall_image, caption="Graphique SHAP - Waterfall Plot", use_column_width=True)


elif menu == 'Visualisation Features':
    st.header("Visualisation des caractéristiques")
    features = st.sidebar.multiselect(
        "Sélectionnez deux caractéristiques :",
        new_clients_df.drop(columns=["SK_ID_CURR"]).columns,
        default=new_clients_df.drop(columns=["SK_ID_CURR"]).columns[:2]
    )
    if len(features) == 2:
        feature1, feature2 = features
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=new_clients_df, x=feature1, y=feature2, alpha=0.6)
        ax.axhline(
            y=new_clients_df.loc[new_clients_df['SK_ID_CURR'] == selected_client_id, feature2].values[0],
            color='red', linestyle='--'
        )
        st.pyplot(fig)
elif menu == 'Description Features':
    st.header("Description des caractéristiques")
    st.dataframe(description_feature_df)
elif menu == 'Feature Importance Globale':
    st.header("Feature Importance Globale")
    global_feature_importances = model.feature_importances_
    global_importance_df = pd.DataFrame({
        'Feature': new_clients_df.drop(columns=["SK_ID_CURR"]).columns,
        'Importance': global_feature_importances
    }).sort_values(by='Importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(global_importance_df['Feature'], global_importance_df['Importance'])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title("Feature Importance Globale")
    ax.invert_yaxis()
    st.pyplot(fig)






