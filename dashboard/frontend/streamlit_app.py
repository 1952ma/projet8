
#sur le terminale 2: streamlit run streamlit_app.py
#si ConnectionError alors Streamlit ne parvient pas à se connecter à API FastAPI => FastAPI n'est pas en cours d'exécution sur le terminale1, il faut l'exécuter

import streamlit as st
import requests
import os

# Récupérer l'URL de l'API depuis les variables d'environnement
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Titre de l'application
st.title("La prédiction du score de crédit")

# Récupérer la liste des clients via l'API FastAPI
st.subheader("Veuillez sélectionner un ID client:")
try:
    response = requests.get(f"{API_URL}/clients")
    response.raise_for_status()  # Vérifier si la requête a réussi
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
        prediction_response.raise_for_status()  # Vérifie si la requête a réussi

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
    
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
    except ValueError as e:
        st.error("Erreur lors de l'analyse de la réponse JSON.")
