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

# Charger le modèle et les données
def load_model_and_data():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model = joblib.load(os.path.join(current_directory, "model.joblib"))
    new_clients_df = pd.read_csv(os.path.join(current_directory, 'df_nouveaux_clients.csv'))
    description_feature_df = pd.read_csv(os.path.join(current_directory, 'description_feature.csv'))
    description_feature_df.columns = ['Feature', 'Description']
    return model, new_clients_df, description_feature_df

# Connexion à l'API FastAPI
def get_client_ids(api_url):
    try:
        response = requests.get(f"{api_url}/clients")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des clients: {e}")
        st.stop()

# Réaliser une prédiction
def get_prediction(api_url, client_id):
    try:
        response = requests.post(f"{api_url}/predict", json={"SK_ID_CURR": client_id})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        return None

# Fonction pour afficher une jauge colorée
def display_gauge(probability, threshold=0.53):
    """
    Affiche une jauge visuelle pour le score de crédit.
    """
    st.subheader("Visualisation du risque : jauge colorée")
    # Construction de la jauge
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(0, threshold, color='green', alpha=0.6, label="Faible risque")
    ax.barh(0, 1 - threshold, left=threshold, color='red', alpha=0.6, label="Haut risque")
    ax.barh(0, probability, color='orange', alpha=0.9, label=f"Client: {probability:.2%}")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, threshold, 1])
    ax.set_xticklabels(['0%', f'{threshold*100:.0f}%', '100%'])
    ax.set_yticks([])
    ax.legend(loc='upper left', fontsize=10)
    plt.title("Risque de défaut de crédit (en probabilité)")
    st.pyplot(fig)

# Charger les données et le modèle
model, new_clients_df, description_feature_df = load_model_and_data()
API_URL = "http://127.0.0.1:8000"

# Titre principal
st.title("Dashboard de prédiction du score de crédit")
st.markdown(
    "Cette application vous permet d'évaluer le risque de crédit des clients. "
    "Elle inclut des outils interactifs d'analyse, de visualisation et d'explicabilité."
)

# Sélection de l'ID client
client_ids = get_client_ids(API_URL)
selected_client_id = st.sidebar.selectbox("Sélectionnez un ID client", client_ids)

# Menu latéral
menu = st.sidebar.radio(
    "Menu",
    ['Prédiction', 'Feature Importance Locale', 'Visualisation Features', 'Description Features', 'Feature Importance Globale']
)

# Option : Prédiction
if menu == 'Prédiction':
    st.header("Prédiction du score de crédit")
    prediction_data = get_prediction(API_URL, selected_client_id)
    if prediction_data:
        probability = prediction_data["probability"]
        prediction_label = prediction_data["prediction_label"]

        # Afficher les résultats
        st.write(f"Probabilité de défaut : **{probability:.2%}**")
        display_gauge(probability)  # Affichage de la jauge

        if prediction_label == 1:
            st.error("Attention : Ce client est susceptible de faire défaut.")
        else:
            st.success("Ce client est susceptible de rembourser son crédit.")

# Option : Feature Importance Locale
elif menu == 'Feature Importance Locale':
    st.header("Feature Importance Locale")
    prediction_data = get_prediction(API_URL, selected_client_id)
    if prediction_data:
        feature_importances = prediction_data["feature_importances"]
        top_features = pd.DataFrame({
            'Feature': new_clients_df.drop(columns=["SK_ID_CURR"]).columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False).head(10)

        # Graphique
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Importance des Features pour le Client {selected_client_id}')
        plt.gca().invert_yaxis()
        st.pyplot(plt)

# Option : Visualisation Features
elif menu == 'Visualisation Features':
    st.header("Visualisation des caractéristiques")
    features = st.sidebar.multiselect(
        "Sélectionnez deux caractéristiques :", 
        new_clients_df.drop(columns=["SK_ID_CURR"]).columns, 
        default=new_clients_df.drop(columns=["SK_ID_CURR"]).columns[:2]
    )
    if len(features) == 2:
        feature1, feature2 = features
        # Graphique : Distribution des caractéristiques
        st.subheader(f"Distribution de {feature1}")
        sns.histplot(new_clients_df, x=feature1, kde=True)
        plt.axvline(new_clients_df.loc[new_clients_df['SK_ID_CURR'] == selected_client_id, feature1].values[0], color='red', linestyle='--')
        st.pyplot(plt)

        st.subheader(f"Distribution de {feature2}")
        sns.histplot(new_clients_df, x=feature2, kde=True)
        plt.axvline(new_clients_df.loc[new_clients_df['SK_ID_CURR'] == selected_client_id, feature2].values[0], color='red', linestyle='--')
        st.pyplot(plt)

# Option : Description Features
elif menu == 'Description Features':
    st.header("Description des caractéristiques")
    selected_feature = st.sidebar.selectbox(
        "Sélectionnez une caractéristique :", 
        description_feature_df['Feature']
    )
    description = description_feature_df.loc[description_feature_df['Feature'] == selected_feature, 'Description'].values[0]
    st.write(f"**{selected_feature}** : {description}")

# Option : Feature Importance Globale
elif menu == 'Feature Importance Globale':
    st.header("Feature Importance Globale")
    global_feature_importances = model.feature_importances_
    global_importance_df = pd.DataFrame({
        'Feature': new_clients_df.drop(columns=["SK_ID_CURR"]).columns,
        'Importance': global_feature_importances
    }).sort_values(by='Importance', ascending=False).head(10)

    # Graphique
    plt.figure(figsize=(10, 6))
    plt.barh(global_importance_df['Feature'], global_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title("Feature Importance Globale")
    plt.gca().invert_yaxis()
    st.pyplot(plt)








