
#sur le terminale 2: streamlit run streamlit_app.py
#si ConnectionError alors Streamlit ne parvient pas à se connecter à API FastAPI => FastAPI n'est pas en cours d'exécution sur le terminale1, il faut l'exécuter

import streamlit as st
import requests
import os
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Récupérer l'URL de l'API depuis les variables d'environnement
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Titre de l'application
st.title("La prédiction du score de crédit")

# Récupérer la liste des clients via l'API FastAPI
st.subheader("Veuillez sélectionner un ID client:")
try:
    response = requests.get(f"{API_URL}/clients")
    response.raise_for_status()
    client_ids = response.json()
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors de la récupération des clients: {e}")
    st.stop()

# Liste déroulante avec les SK_ID_CURR
selected_client_id = st.selectbox("SK_ID_CURR (ID Clients)", client_ids)

# Bouton pour lancer la prédiction
if st.button("Réaliser une prédiction"):
    try:
        prediction_response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_client_id})
        prediction_response.raise_for_status()
        prediction_data = prediction_response.json()

        if 'probability' in prediction_data:
            probability = prediction_data["probability"]
        else:
            st.error("La clé 'probability' est introuvable dans la réponse de l'API.")
            st.stop()

        if "error" in prediction_data:
            st.error(prediction_data["error"])
        else:
            prediction_label = prediction_data["prediction_label"]
            feature_importance_local = prediction_data["feature_importance_local"]
            global_feature_importance = prediction_data["global_feature_importance"]

            st.subheader("Résultats de la prédiction")
            st.write(f"Probabilité de défaut: **{probability * 100:.2f}%**")
            st.write(f"Décision: **{'Refus' if prediction_label else 'Acceptation'}**")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Probabilité de défaut (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if probability > 0.5 else "green"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgreen"},
                        {"range": [50, 100], "color": "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig)

            st.subheader("Feature Importance Locale")
            fig_local = px.bar(pd.DataFrame(feature_importance_local), x="feature", y="importance", title="Importance des Features pour ce client")
            st.plotly_chart(fig_local)

            st.subheader("Feature Importance Globale")
            fig_global = px.bar(pd.DataFrame(global_feature_importance), x="feature", y="importance", title="Importance Globale des Features")
            st.plotly_chart(fig_global)

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
    except ValueError as e:
        st.error("Erreur lors de l'analyse de la réponse JSON.")






