import sys
import os

# Ajout du chemin vers le module compilé (si nécessaire, selon où on lance le script)
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

try:
    import sp500_rf_cpp # Notre module C++
except ImportError:
    print("Erreur: Le module C++ n'est pas trouvé. Avez-vous compilé avec CMake ?")
    sys.exit(1)

from data_loader import SP500Loader

def main():
    # 1. Chargement des données
    # 2. Préparation (Features engineering)
    # 3. Instanciation du RandomForest C++
    # 4. Entraînement et Prédiction
    pass

if __name__ == "__main__":
    main()