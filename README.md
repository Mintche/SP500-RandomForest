# SP500-RandomForest

Projet de prédiction du S&P500 utilisant un Random Forest codé en C++ et intégré via pybind11.

## Architecture

*   **src/** : Code C++ (RandomForest, DecisionTree) et bindings.
*   **include** : Header C++ (RandomForest, DecisionTree).
*   **python/** : Code Python (Chargement de données, Main).
*   **build/** : Dossier de compilation (généré par CMake).

## Installation

```bash
pip install -r requirements.txt
mkdir build && cd build
cmake ..
make
```