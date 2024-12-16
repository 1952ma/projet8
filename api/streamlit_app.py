# cd api

#sur le terminale 2: streamlit run streamlit_app.py
#si ConnectionError alors Streamlit ne parvient pas à se connecter à API FastAPI => FastAPI n'est pas en cours d'exécution sur le terminale1, il faut l'exécuter


import streamlit as st
import requests
import pandas as pd
import os

# Chargement des données des nouveaux clients (CSV)

current_directory = os.path.dirname(os.path.realpath(__file__))
new_clients_df = pd.read_csv(os.path.join(current_directory, 'df_nouveaux_clients.csv'))

# Charger les données des nouveaux clients depuis FastAPI
API_URL = "http://127.0.0.1:8000"

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
        # Requête POST pour obtenir la prédiction
        prediction_response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_client_id})
        prediction_response.raise_for_status()

        # Extraction des données de la réponse JSON
        prediction_data = prediction_response.json()

        # Afficher les résultats
        if "error" in prediction_data:
            st.error(prediction_data["error"])
        else:
            probability = prediction_data["probability"]
            prediction_label = prediction_data["prediction_label"]

            # Afficher la probabilité
            st.success(f"La probabilité de défaut pour ce client est de: {probability:.2%}")

            # Afficher l'évaluation du risque
            if prediction_label == 1:
                st.markdown("<span style='color:red;'>Attention : Ce client est susceptible de <b>faire défaut</b> sur son crédit (Classe = 1)</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:green;'>Ce client est susceptible de <b>rembourser</b> son crédit (Classe = 0)</span>", unsafe_allow_html=True)

            # Fonctionnalité de jauge colorée du score
            st.subheader("Score détaillé avec jauge colorée")
            color = 'green' if prediction_data['probability'] < 0.53 else 'red'
            st.markdown(f'<div style="width: 100%; height: 30px; background: linear-gradient(to right, green, red);"></div>', unsafe_allow_html=True)

            # Fonctionnalité pour Feature Importance locale
            st.subheader("Feature Importance Locale")
            local_feature_importance = {}  # Charger les importances locales depuis FastAPI ou calculer localement
            st.bar_chart(local_feature_importance)

            # Sélection de deux Features pour visualisation
            st.subheader("Distribution des Features Sélectionnés")
            features = new_clients_df.columns.drop('SK_ID_CURR').tolist()
            feature1 = st.selectbox("Sélectionner la première feature", features)
            feature2 = st.selectbox("Sélectionner la deuxième feature", features)

            # Analyse bi-variée entre les deux features sélectionnés
            st.subheader("Analyse Bi-Variée entre deux Features")
            feature1_values = new_clients_df[feature1]
            feature2_values = new_clients_df[feature2]
            st.write(f"Visualisation du dégradé selon {feature1} et {feature2}")
            # Afficher un graphique bi-varié avec un dégradé de couleur

            # Feature Importance Globale
            st.subheader("Feature Importance Globale")
            global_feature_importance = {}  # Charger les importances globales depuis FastAPI ou calculer localement
            st.bar_chart(global_feature_importance)

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
    except ValueError as e:
        st.error("Erreur lors de l'analyse de la réponse JSON.")



