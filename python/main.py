import numpy as np
import matplotlib.pyplot as plt
import RandomForestcpp
from data_loader import SP500Loader

def main():
    # 1. Chargement des données
    HORIZON = 5
    loader = SP500Loader(start_date="2010-01-01", end_date="2023-12-31", forecast_horizon=HORIZON)
    df = loader.fetch_data()
    
    if df.empty:
        print("Erreur: Aucune donnée récupérée.")
        return

    # 2. Préparation
    X, y = loader.prepare_features_targets(df)
    print(f"Données prêtes: {X.shape[0]} échantillons avec {X.shape[1]} features.")

    # Split Train/Test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 3. Instanciation du RandomForest C++
    print("Initialisation du RandomForest C++...")
    rf = RandomForestcpp.RandomForest(
        n_trees=500, 
        max_depth=20, 
        min_samples_split=10,
        task_type=RandomForestcpp.TaskType.REGRESSION
    )

    # 4. Entraînement et Prédiction
    print("Entraînement du modèle...")
    rf.fit(X_train, y_train)

    print("Prédiction sur le jeu de test...")
    predictions = rf.predict(X_test)
    predictions = np.array(predictions)
    predictions = predictions.flatten()

    # Reconstruction des prix

    last_close_prices = X_test[:, 0]
    
    predicted_prices = last_close_prices * (1 + predictions)
    real_prices = last_close_prices * (1 + y_test)

    # Évaluation (RMSE sur les prix reconstruits)
    naive_predictions = np.zeros_like(y_test)
    naive_predicted_prices = last_close_prices * (1 + naive_predictions)
    
    rmse = np.sqrt(np.mean((predicted_prices - real_prices)**2))
    rmse_naive = np.sqrt(np.mean((naive_predicted_prices - real_prices)**2))

    print(f"RMSE du Modèle Naïf (Rendement=0) sur {HORIZON} jours: {rmse_naive:.4f}")
    print(f"RMSE du Random Forest sur {HORIZON} jours: {rmse:.4f}")

    # Visualisation des résultats
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label='Réel (Prix)', color='blue', alpha=0.6)
    plt.plot(predicted_prices, label='Prédiction (Prix reconstruit)', color='red', alpha=0.6, linestyle='--')
    plt.title(f'Prédiction S&P 500 (RMSE: {rmse:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig("prediction_sp500.png")
    print("Graphique sauvegardé dans 'prediction_sp500.png'")

if __name__ == "__main__":
    main()