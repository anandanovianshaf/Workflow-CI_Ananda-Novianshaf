# Pakai base image Python 3.10 yang sudah stabil
FROM python:3.10-slim

# Set folder kerja di dalam container
WORKDIR /app

# Install dependency yang dibutuhin
RUN pip install mlflow pandas scikit-learn

# Copy semua file project lo ke dalam container
COPY . /app

# Buka port 5000 buat MLflow
EXPOSE 5000

# Perintah buat ngejalanin modelnya pas container hidup
CMD ["mlflow", "models", "serve", "-m", "model_artifact", "-h", "0.0.0.0", "-p", "5000"]