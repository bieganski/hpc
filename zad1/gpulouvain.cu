#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

static float MIN_GAIN;
static char *FILE_PATH;
static bool VERBOSE = 0;

using node_t = uint32_t;

static node_t* V; // vertices
static node_t* E; // edges

int parse_args (int argc, char **argv) {
    int index;
    int gflag = 0, fflag = 0;
    int c;

    opterr = 0;

    while ((c = getopt (argc, argv, "f:g:v")) != -1)
        switch (c) {
            case 'v':
                VERBOSE = 1;
                break;
            case 'f':
                fflag = 1;
                FILE_PATH = optarg;
                break;
            case 'g':
                gflag = 1;
                try {
                    MIN_GAIN = std::stof(std::string(optarg));
                } catch (std::invalid_argument) {
                    fprintf (stderr, "Min Gain must be proper float number!\n");
                    return 1;
                }
                break;
            case '?':
                if (optopt == 'f' || optopt == 'g')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return 1;
            default:
                abort ();
            }

    if (!fflag) {
        fprintf (stderr, "Filename option -f is required!");
        return 1;
    } else if(!gflag) {
        fprintf (stderr, "Min gain option -g is required!");
        return 1;
    }

    return 0;
}

std::ifstream get_input_content() {
    try {
        std::ifstream file_stream{FILE_PATH};
        // std::string file_content((std::istreambuf_iterator<char>(file_stream)),
        //                         std::istreambuf_iterator<char>());

        // file_stream.close();
        
        // if (file_content.size() == 0) {
        //     std::cerr << "Error reading input file" + std::string(FILE_PATH);
        //     exit(1);
        // }

        return file_stream;

    } catch (const char * f) {
        std::cerr << "Error reading input file" + std::string(FILE_PATH);
        exit(1);
    }
}


void parse_inut_graph(std::ifstream content) {
    std::stringstream ss;
    ss << content.rdbuf();    
    content.close();
    std::string line;

    do {
        std::getline(ss, line, '\n'); 
    } while (line[0] == '%'); // comments

    // first line is special, contains num of edges and max vertex num 
    std::stringstream tmp_ss(line);
    node_t v1max, v2max;
    uint64_t EDGES;
    tmp_ss >> v1max;
    tmp_ss >> v2max;
    tmp_ss >> EDGES;

    assert(v1max == v2max); // TODO wywalić i zastąpić max(...);
    
    node_t N = v1max + 1;
    // compressed neighbour list requires creating intermediate representation
    std::vector<std::vector<uint32_t>> tmpG(N);

    // read rest of lines
    uint32_t v1, v2;
    float w = 1;
    bool weights = true; // if false then assume all weights 1
    while(std::getline(ss, line, '\n')){

        if (line.size() == 0)
            continue;

        tmp_ss = std::stringstream();
        tmp_ss << line;
        
        tmp_ss >> v1;
        tmp_ss >> v2;

        // code below is kinda hacky, but seems to be fast
        if (weights == false) {
            w = 1;
        } else {
            if (!(tmp_ss >> w)) {
                weights = false;
                w = 1;
            }
        }

        tmpG[v1].push_back(v2);
        tmpG[v2].push_back(v1);
    }
    
    // assumption: no multi-edges in input graph

    V = (node_t*) malloc(sizeof(node_t) * (N + 1));
    E = (node_t*) malloc(sizeof(node_t) * (2 * EDGES));

    node_t act = 0;
    for (node_t i = 1; i < N; i++) {
        V[i] = act;
        std::copy(tmpG[i].begin(), tmpG[i].end(), E + act);
        act += tmpG[i].size();
        V[i + 1] = act;
    }
}

int main(int argc, char **argv) {
    if (parse_args(argc, argv)) {
        exit(1);
    }
    
    parse_inut_graph(get_input_content());

    return 0;
}