# cd api

#sur le terminale 2: streamlit run streamlit_app.py
#si ConnectionError alors Streamlit ne parvient pas à se connecter à API FastAPI => FastAPI n'est pas en cours d'exécution sur le terminale1, il faut l'exécuter


import streamlit as st
import requests
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt


current_directory = os.path.dirname(os.path.realpath(__file__))

# Chargement des données des nouveaux clients (CSV)
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
            #st.success(f"La probabilité de défaut pour ce client est de: {probability:.2%}")

            # Afficher l'évaluation du risque
            if prediction_label == 1:
                st.markdown("<span style='color:red;'>Attention : ce client est susceptible de <b>faire défaut</b> sur son crédit.</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:green;'>ce client est susceptible de <b>rembourser</b> son crédit. </span>", unsafe_allow_html=True)
            
            # Définir le seuil pour changer la couleur 
            threshold = 0.53 
            score_color = 'green' if probability < threshold else 'red' 
            
            # Fonctionnalité de jauge colorée du score avec un curseur
            st.subheader("Score détaillé avec jauge colorée")

            # Définir le seuil pour changer la couleur
            threshold = 0.53
            score_color = 'green' if probability < threshold else 'red'

            # Afficher la barre de progression
            st.markdown(
                f"<div style='position: relative; width: 100%; height: 30px; background: linear-gradient(to right, green {threshold*100}%, red {threshold*100}%);'></div>",
                unsafe_allow_html=True
            )
            
            # Saut de ligne entre les jauges
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(
                f"<div style='position: relative; width: {probability*100}%; height: 30px; background-color: {score_color}; text-align: center; color: white;'>"
                f"{'<b>Faible</b> risque' if score_color == 'green' else '<b>Haut</b> risque'}: la probabilité de défaut pour ce client : {probability:.2%}</div>", 
                unsafe_allow_html=True
            )
            # Visualiser l'importance des features pour ce client
            feature_importances = prediction_data["feature_importances"]
            df_columns = new_clients_df.drop(columns=["SK_ID_CURR"]).columns

            top_features = pd.DataFrame({
                'Feature': df_columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            plt.barh(top_features['Feature'], top_features['Importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Importances des Features pour le Client {selected_client_id}')
            plt.gca().invert_yaxis()
            st.pyplot(plt)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
    except ValueError as e:
        st.error("Erreur lors de l'analyse de la réponse JSON.")




