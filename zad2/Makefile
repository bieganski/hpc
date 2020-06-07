SRC_DIR := src
SRC_FILES := $(shell ls -1)
OBJ_DIR := build
OBJ_FILES := $(shell ls src/*.cpp | sed -e "s/.cpp/.o/g" | sed -e "s~$(SRC_DIR)/~$(OBJ_DIR)/~g")

CXX := mpic++
CXXFLAGS := -g
# LDFLAGS :=

IGNORE := $(shell mkdir -p $(OBJ_DIR))

all: main

main: $(OBJ_FILES)
	mkdir -p $(OBJ_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_FILES)
