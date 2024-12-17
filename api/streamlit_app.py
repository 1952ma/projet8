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

    # Chargement de la description des features
    description_feature_df = pd.read_csv(
        os.path.join(current_directory, 'description_feature_cleaned.csv'),
        sep=',',
        quotechar='"',
        skipinitialspace=True
    )

    # Nom des colonnes
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
threshold = 0.53  # Seuil à 53%

# Construction de la jauge avec marqueurs
fig, ax = plt.subplots(figsize=(8, 1))
ax.barh(0, threshold, color='green', alpha=0.9, label="Faible risque")
ax.barh(0, 1 - threshold, left=threshold, color='red', alpha=0.9, label="Haut risque")
ax.barh(0, threshold, color='green', alpha=0.9)

# Ajout des marqueurs à 0%, 53%, et 100%
ax.barh(0, 1, color='none', edgecolor='black')
ax.set_xlim(0, 1)
ax.set_xticks([0, threshold, 1])
ax.set_xticklabels(['0%', f'{threshold*100:.0f}%', '100%'])
ax.set_yticks([])
ax.legend(loc='upper left', fontsize=10)
plt.title("Risque de défaut de crédit (en probabilité)")

st.pyplot(fig)


# --- Sélection de l'ID client ---
client_ids = get_client_ids(API_URL)
selected_client_id = st.selectbox("Liste des clients :", client_ids)

# --- Prédiction pour le client sélectionné ---
prediction_data = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if st.button("Annalyse"):
    st.session_state.prediction_data = get_prediction(API_URL, selected_client_id)



# Affichage permanent de la jauge client avec une taille agrandie
if st.session_state.prediction_data:
    probability = st.session_state.prediction_data["probability"]
    threshold = 0.53  # Seuil défini à 53%
    score_color = 'green' if probability < threshold else 'red'
    
    # Construction de la jauge agrandie
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.barh(0, probability, color=score_color, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, threshold, 1])
    ax.set_xticklabels(['0%', f'{threshold*100:.0f}%', '100%'])
    ax.set_yticks([])
    ax.set_title(f"Le client : {selected_client_id} => Risque de défaut de crédit = {probability:.2%}")
    
    st.pyplot(fig)



# --- Menu latéral ---
menu = st.sidebar.radio(
    "Menu",
    ['Selectionner :', 'Feature Importance Locale', 'Description Features', 'Feature Importance Globale', 'Visualiser la distribution des features', 'Exploration des données']
)

if menu == 'Selectionner :':
    st.header("")

# --- Feature Importance Locale ---
elif menu == 'Feature Importance Locale':
    st.header("Feature Importance Locale")
    shap_waterfall = get_shap_graph(selected_client_id)
    waterfall_image = Image.open(BytesIO(base64.b64decode(shap_waterfall)))
    st.image(waterfall_image, caption="Graphique SHAP - Waterfall Plot", use_column_width=True)

elif menu == 'Description Features':
    st.header("Feature Importance Locale")
    shap_waterfall = get_shap_graph(selected_client_id)
    waterfall_image = Image.open(BytesIO(base64.b64decode(shap_waterfall)))
    st.image(waterfall_image, caption="Graphique SHAP - Waterfall Plot", use_column_width=True)

    st.header("Description des caractéristiques")
    st.dataframe(description_feature_df)

elif menu == 'Feature Importance Globale':
    st.header("Feature Importance Locale")
    shap_waterfall = get_shap_graph(selected_client_id)
    waterfall_image = Image.open(BytesIO(base64.b64decode(shap_waterfall)))
    st.image(waterfall_image, caption="Graphique SHAP - Waterfall Plot", use_column_width=True)
    
    st.header("Feature Importance Globale")
    global_feature_importances = model.feature_importances_
    global_importance_df = pd.DataFrame({
        'Feature': new_clients_df.drop(columns=["SK_ID_CURR"]).columns,
        'Importance': global_feature_importances
    }).sort_values(by='Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(global_importance_df['Feature'], global_importance_df['Importance'])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title("Feature Importance Globale")
    ax.invert_yaxis()
    st.pyplot(fig)

elif menu == 'Visualiser la distribution des features':
    st.header("Visualisation des caractéristiques")

    # Sélection des features
    features = st.sidebar.multiselect(
        "Sélectionnez deux caractéristiques :",
        new_clients_df.drop(columns=["SK_ID_CURR"]).columns
    )

    # Vérification que deux caractéristiques sont sélectionnées
    if len(features) == 2:
        feature1, feature2 = features

        # Ajout des classes prédictes pour visualisation
        if "classe_predite" not in new_clients_df.columns:
            new_clients_df['classe_predite'] = model.predict(new_clients_df.drop(columns=["SK_ID_CURR"]))

        # Définir les couleurs pour chaque classe
        couleurs = {0: 'green', 1: 'red'}  # Classe 0 = faible risque (vert), Classe 1 = haut risque (rouge)

        # Visualisation de la distribution de la première caractéristique
        st.subheader(f"Distribution de la caractéristique : {feature1}")
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        # Boucle pour personnaliser les couleurs et les labels dans la légende
        for classe in new_clients_df['classe_predite'].unique():
            label = f"Classe {classe} ({'Faible risque' if classe == 0 else 'Haut risque'})"
            subset = new_clients_df[new_clients_df['classe_predite'] == classe]
            sns.kdeplot(
                data=subset,
                x=feature1,
                fill=True,
                alpha=0.5,
                label=label,
                color=couleurs[classe],
                ax=ax1
            )

        # Ajout de la valeur du client
        client_value1 = new_clients_df.loc[new_clients_df['SK_ID_CURR'] == selected_client_id, feature1].values[0]
        ax1.axvline(client_value1, color='blue', linestyle='--', label=f"Valeur client ({client_value1:.2f})")

        # Ajustements
        ax1.set_title(f"Distribution de {feature1} selon les classes prédictes")
        ax1.set_xlabel(feature1)
        ax1.legend()
        st.pyplot(fig1)

        # Visualisation de la distribution de la deuxième caractéristique
        st.subheader(f"Distribution de la caractéristique : {feature2}")
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Boucle pour personnaliser les couleurs et les labels dans la légende
        for classe in new_clients_df['classe_predite'].unique():
            label = f"Classe {classe} ({'Faible risque' if classe == 0 else 'Haut risque'})"
            subset = new_clients_df[new_clients_df['classe_predite'] == classe]
            sns.kdeplot(
                data=subset,
                x=feature2,
                fill=True,
                alpha=0.5,
                label=label,
                color=couleurs[classe],
                ax=ax2
            )

        # Ajout de la valeur du client
        client_value2 = new_clients_df.loc[new_clients_df['SK_ID_CURR'] == selected_client_id, feature2].values[0]
        ax2.axvline(client_value2, color='blue', linestyle='--', label=f"Valeur client ({client_value2:.2f})")

        # Ajustements
        ax2.set_title(f"Distribution de {feature2} selon les classes prédictes")
        ax2.set_xlabel(feature2)
        ax2.legend()
        st.pyplot(fig2)

elif menu == 'Exploration des données':
    st.header("Exploration des données")

    # Sélection d'une feature pour l'histogramme
    feature_hist = st.sidebar.selectbox(
        "Sélectionnez une caractéristique pour l'histogramme :",
        new_clients_df.drop(columns=["SK_ID_CURR"]).columns
    )

    # Histogramme interactif
    st.subheader(f"Distribution de la caractéristique : {feature_hist}")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.histplot(data=new_clients_df, x=feature_hist, kde=True, bins=30)
    ax1.set_title(f"Distribution de {feature_hist}")
    st.pyplot(fig1)

    # Diagramme de corrélation
    st.subheader("Diagramme de corrélation entre les caractéristiques")
    corr_matrix = new_clients_df.drop(columns=["SK_ID_CURR"]).corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    ax2.set_title("Corrélation entre les caractéristiques")
    st.pyplot(fig2)













