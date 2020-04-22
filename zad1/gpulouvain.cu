#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "hasharray.h"
#include "utils.h"

using namespace std;

float MIN_GAIN;
char *FILE_PATH;
bool VERBOSE = 0;

static node_t* V; // vertices
static node_t* E; // edges

int main(int argc, char **argv) {
    if (parse_args(argc, argv)) {
        exit(1);
    }
    
    parse_inut_graph(get_input_content());

    return 0;
}