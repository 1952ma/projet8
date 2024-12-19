# Utilise une image Python légère
FROM python:3.12-slim

# Définit le répertoire de travail
WORKDIR /frontend

# Copie le fichier des dépendances du frontend
COPY frontend/requirements.txt /requirements.txt

# Installe les dépendances du frontend
RUN pip install --no-cache-dir -r /requirements.txt

# Copie le reste des fichiers du frontend
COPY frontend/ .

# Expose le port sur lequel Streamlit écoutera
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

