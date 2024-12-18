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
import plotly.express as px 
import shap

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
    ['Selectionner :', 'Feature Importance Locale', 'Feature Importance Globale', 'Visualiser la distribution des features', 'Exploration des données', 'Description Features']
)

if menu == 'Selectionner :':
    st.header("")

# --- Feature Importance Locale ---
elif menu == 'Feature Importance Locale':
    if st.session_state.prediction_data:
        st.header(f"Analyse locale pour le client : {selected_client_id}")
        
        # Sélectionner les données du client à partir de son ID
        client_data = new_clients_df[new_clients_df["SK_ID_CURR"] == selected_client_id]
        
        if client_data.empty:
            st.error("Données du client introuvables.")
        else:
            # Créer un explainer SHAP pour LightGBM
            explainer = shap.Explainer(model, new_clients_df)
            
            # Calculer les valeurs SHAP pour le client sélectionné
            shap_values = explainer(client_data)
            
            # Afficher le graphique SHAP Waterfall pour le client
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(shap_values[0], show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage du graphique SHAP : {e}")
    else:
        st.warning("Veuillez d'abord effectuer une analyse pour un client spécifique.")

    st.header("Description des caractéristiques")
    st.dataframe(description_feature_df)

elif menu == 'Feature Importance Globale':
    
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

    st.header("Description des caractéristiques")
    st.dataframe(description_feature_df)

    if st.session_state.prediction_data:
        st.header(f"Analyse locale pour le client : {selected_client_id}")
        
        # Sélectionner les données du client à partir de son ID
        client_data = new_clients_df[new_clients_df["SK_ID_CURR"] == selected_client_id]
        
        if client_data.empty:
            st.error("Données du client introuvables.")
        else:
            # Créer un explainer SHAP pour LightGBM
            explainer = shap.Explainer(model, new_clients_df)
            
            # Calculer les valeurs SHAP pour le client sélectionné
            shap_values = explainer(client_data)
            
            # Afficher le graphique SHAP Waterfall pour le client
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(shap_values[0], show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage du graphique SHAP : {e}")
    else:
        st.warning("Veuillez d'abord effectuer une analyse pour un client spécifique.")

    st.header("Description des caractéristiques")
    st.dataframe(description_feature_df)

elif menu == 'Visualiser la distribution des features':
    st.header("Visualisation des caractéristiques")

    # Sélection des 50 premières features mentionnées
    features = st.sidebar.multiselect(
        "Sélectionnez deux caractéristiques :",
        ['EXT_SOURCE_2', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'PAYMENT_RATE', 'DAYS_EMPLOYED',
         'INSTAL_DPD_MEAN', 'AMT_ANNUITY', 'DAYS_BIRTH', 'INSTAL_AMT_PAYMENT_SUM', 'CODE_GENDER',
         'INSTAL_AMT_PAYMENT_MIN', 'PREV_CNT_PAYMENT_MEAN', 'AMT_CREDIT', 'ACTIVE_DAYS_CREDIT_MAX',
         'OWN_CAR_AGE', 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'APPROVED_CNT_PAYMENT_MEAN', 'AMT_GOODS_PRICE',
         'ANNUITY_INCOME_PERC', 'NAME_EDUCATION_TYPE_Higher_education', 'POS_MONTHS_BALANCE_SIZE',
         'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'APPROVED_AMT_DOWN_PAYMENT_MAX', 'POS_SK_DPD_DEF_MEAN',
         'PREV_APP_CREDIT_PERC_MIN', 'DAYS_EMPLOYED_PERC', 'INSTAL_PAYMENT_PERC_MEAN',
         'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
         'BURO_DAYS_CREDIT_MAX', 'PREV_APP_CREDIT_PERC_MEAN', 'ACTIVE_DAYS_CREDIT_ENDDATE_MEAN',
         'ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN', 'BURO_CREDIT_TYPE_Microloan_MEAN',
         'BURO_AMT_CREDIT_SUM_MAX', 'NAME_CONTRACT_TYPE_Cash_loans', 'PREV_NAME_YIELD_GROUP_high_MEAN',
         'ACTIVE_DAYS_CREDIT_ENDDATE_MAX', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'CLOSED_AMT_CREDIT_SUM_SUM',
         'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_CREDIT_ACTIVE_Closed_MEAN', 'INSTAL_AMT_PAYMENT_MEAN',
         'INSTAL_DAYS_ENTRY_PAYMENT_SUM', 'BURO_DAYS_CREDIT_MEAN', 'INSTAL_DBD_SUM',
         'FLAG_DOCUMENT_3', 'BURO_CREDIT_TYPE_Mortgage_MEAN', 'NAME_FAMILY_STATUS_Married',
         'REGION_RATING_CLIENT_W_CITY'],
        help="Sélectionnez exactement deux caractéristiques"
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

    # Liste des 20 premières features
    top_20_features = [
        'EXT_SOURCE_2', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'PAYMENT_RATE', 'DAYS_EMPLOYED',
        'INSTAL_DPD_MEAN', 'AMT_ANNUITY', 'DAYS_BIRTH', 'INSTAL_AMT_PAYMENT_SUM', 'CODE_GENDER',
        'INSTAL_AMT_PAYMENT_MIN', 'PREV_CNT_PAYMENT_MEAN', 'AMT_CREDIT', 'ACTIVE_DAYS_CREDIT_MAX',
        'OWN_CAR_AGE', 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'APPROVED_CNT_PAYMENT_MEAN', 'AMT_GOODS_PRICE',
        'ANNUITY_INCOME_PERC', 'NAME_EDUCATION_TYPE_Higher_education'
    ]

    # Affichage de chaque graphique interactif
    for feature in top_20_features:
        st.subheader(f"Distribution de la caractéristique : {feature}")
        
        # Utilisation de st.plotly_chart pour un graphique interactif
        fig = px.histogram(new_clients_df, x=feature, marginal="box", title=f"Distribution de {feature}")
        st.plotly_chart(fig)

elif menu == 'Description Features':
    st.header("Description des caractéristiques")
    st.dataframe(description_feature_df)












