cmake_minimum_required(VERSION 3.10)
project(sp500_rf_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Trouver pybind11 (installé via pip)
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# Création du module Python
pybind11_add_module(sp500_rf_cpp src/bindings.cpp src/RandomForest.cpp src/DecisionTree.cpp)
target_include_directories(sp500_rf_cpp PRIVATE src)