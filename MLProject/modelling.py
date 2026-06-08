import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

mlflow.set_experiment("CI_Pipeline_Network_Flow")

with mlflow.start_run() as run:
    print("Loading data...")
    df = pd.read_csv('dataset_ready.csv')
    X = df.drop(['f', 'f_scaled'], axis=1)
    y = df['f_scaled']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Log model ke folder mlruns
    mlflow.sklearn.log_model(rf, "model")
    
    # Trik: Ambil absolute local path dari artifact URI
    # run.info.artifact_uri bentuknya "file:///home/runner/.../artifacts"
    artifact_path = run.info.artifact_uri.replace("file://", "")
    model_path = os.path.join(artifact_path, "model")
    
    # Simpan path asli ini ke file teks (bukan cuma Run ID)
    with open("model_path.txt", "w") as f:
        f.write(model_path)
        
    print(f"Model berhasil dilatih dan tersimpan di: {model_path}")
