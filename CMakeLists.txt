cmake_minimum_required(VERSION 3.12)
project(SP500_RandomForest)

# Le projet utilise C++17 (pour std::unique_ptr, les templates, etc.)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Recherche de Python et de pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# On demande à pybind11 son chemin de configuration CMake
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)

# Recherche d'OpenMP
find_package(OpenMP REQUIRED)

# Spécifie où se trouvent tes fichiers headers (.hpp)
include_directories(include)

# Liste des fichiers sources à compiler
set(SOURCES
    src/DecisionTree.cpp
    src/RandomForest.cpp
    src/bindings.cpp
)

# Création du module Python
# Le nom 'RandomForestcpp' doit correspondre exactement à celui dans PYBIND11_MODULE
pybind11_add_module(RandomForestcpp ${SOURCES})

# Liaison avec OpenMP (gère automatiquement les flags de compilation et de link)
target_link_libraries(RandomForestcpp PRIVATE OpenMP::OpenMP_CXX)

# Optimisations de performance
if(MSVC)
    target_compile_options(RandomForestcpp PRIVATE /O2 /fp:fast)
else()
    target_compile_options(RandomForestcpp PRIVATE -O3 -march=native -ffast-math)
endif()

# Installation du module (nécessaire pour scikit-build-core / pip install .)
install(TARGETS RandomForestcpp DESTINATION .)