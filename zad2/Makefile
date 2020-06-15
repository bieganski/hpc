SRC_DIR := src
OBJ_DIR := build
OBJ_FILES := $(shell ls src/*.cpp | sed -e "s/.cpp/.o/g" | sed -e "s~$(SRC_DIR)/~$(OBJ_DIR)/~g")

CXX := mpic++
CXXFLAGS := -g -Wmaybe-uninitialized -O3

IGNORE := $(shell mkdir -p $(OBJ_DIR))

all: body3

body3: $(OBJ_FILES)
	mkdir -p $(OBJ_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_FILES)
