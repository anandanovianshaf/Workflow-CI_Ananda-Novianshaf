import pandas as pd
import mlflow
import mlflow.sklearn
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
    
    # Log model ke folder default 'mlruns'
    mlflow.sklearn.log_model(rf, "model")
    
    # Simpan Run ID ke sebuah file teks biar bisa dibaca sama GitHub Actions
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
        
    print(f"Model berhasil dilatih dengan Run ID: {run.info.run_id}")