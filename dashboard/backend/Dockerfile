# Choisir une image de base Python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires (code source et dépendances)
COPY backend/requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source du backend
COPY backend /app

# Exposer le port de l'application FastAPI
EXPOSE 8000

# Lancer l'application FastAPI avec uvicorn
CMD ["uvicorn", "fast_app:app", "--host", "0.0.0.0", "--port", "8000"]
