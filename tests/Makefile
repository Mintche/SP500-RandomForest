CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -I../include
CPPFLAGS = -MMD -MP

SRC_DIR = ../src
OBJ_DIR = obj

# Sources du projet et du test
SOURCES = $(SRC_DIR)/DecisionTree.cpp $(SRC_DIR)/RandomForest.cpp
TEST_SRC = test_main.cpp

# Transformation des sources en objets
OBJS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TEST_OBJ = $(OBJ_DIR)/test_main.o

# Fichiers de dépendances (.d)
DEPS = $(OBJS:.o=.d) $(TEST_OBJ:.o=.d)

TARGET = run_tests

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/test_main.o: $(TEST_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

test: all
	./$(TARGET)

clean:
	$(RM) -r $(OBJ_DIR) $(TARGET)

-include $(DEPS)
