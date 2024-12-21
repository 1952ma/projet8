# Utiliser une image de base
FROM node:16-alpine

# Définir le répertoire de travail
WORKDIR /app/frontend

# Copier les fichiers nécessaires
COPY frontend/package.json /app/frontend/package.json
COPY frontend/package-lock.json /app/frontend/package-lock.json

# Installer les dépendances
RUN npm install

# Copier le reste des fichiers
COPY frontend/ /app/frontend/

# Expose le port sur lequel Streamlit écoutera
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]




