# Utilise une image Python légère
FROM python:3.12-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie le fichier des dépendances
COPY frontend/requirements.txt app/requirements.txt

# Installe les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copie le reste des fichiers dans le conteneur
COPY frontend/ /app

# Expose le port sur lequel Streamlit écoutera
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]




