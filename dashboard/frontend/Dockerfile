# Choisir une image de base Python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt
COPY frontend/requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source du frontend
COPY frontend /app

# Exposer le port de l'application Streamlit
EXPOSE 8501

# Lancer l'application Streamlit
CMD ["streamlit", "run", "streamlit_app.py"]




