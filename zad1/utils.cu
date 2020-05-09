#include <cstdint>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <set>
#include <iterator>
#include <algorithm>

#include "utils.h"

static uint32_t* V; // vertices
static uint32_t* E; // edges
static float* W; // weights
static float* k; // sum of weights per node
static float m = 0; // total sum of weights

__host__
void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit( EXIT_FAILURE );
    }
}


__host__ 
int parse_args (int argc, char **argv) {
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

__host__ 
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


__host__ 
ret_t parse_inut_graph(std::ifstream content) {
    std::stringstream ss;
    ss << content.rdbuf();    
    content.close();
    std::string line;

    do {
        std::getline(ss, line, '\n'); 
    } while (line[0] == '%'); // comments

    // first line is special, contains num of edges and max vertex num 
    std::stringstream tmp_ss(line);
    uint32_t v1max, v2max;
    uint32_t EDGES;
    tmp_ss >> v1max;
    tmp_ss >> v2max;
    tmp_ss >> EDGES;

    assert(v1max == v2max); // TODO wywalić i zastąpić max(...);
    
    uint32_t N = v1max + 1;

    // compressed neighbour list requires creating intermediate representation
    std::vector<std::vector<uint32_t>> tmpG(N);
    std::vector<std::vector<float>> tmpW(N);


    // assumption: no multi-edges in input graph

    HANDLE_ERROR(cudaHostAlloc((void**)&V, sizeof(uint32_t) * (N+1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&E, sizeof(uint32_t) * (2*EDGES), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&W, sizeof(float) * (2*EDGES), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&k, sizeof(float) * N, cudaHostAllocDefault));

    std::memset(k, '\0', sizeof(float) * N);

    // read rest of lines
    uint32_t v1, v2;
    float w = 1.0;
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
            w = 1.0;
        } else {
            if (!(tmp_ss >> w)) {
                weights = false;
                w = 1.0;
            }
        }

        tmpG[v1].push_back(v2);
        tmpW[v1].push_back(w);
        k[v1] += w;
        
        if (v1 != v2) {
            tmpG[v2].push_back(v1);
            tmpW[v2].push_back(w);
            k[v2] += w;
        }
        m += w;
    }

    uint32_t act = 0;
    for (uint32_t i = 1; i < N; i++) {
        V[i] = act;
        std::copy(tmpG[i].begin(), tmpG[i].end(), E + act);
        std::copy(tmpW[i].begin(), tmpW[i].end(), W + act);
        act += tmpG[i].size();
        V[i + 1] = act;
    }

    return std::make_tuple(N - 1, V, E, W, k, m);
}


// https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/
__host__ __device__ 
uint32_t next_2_pow(uint32_t n) { 
    assert(n > 1);
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;

    for (uint32_t i = 0; i <= 31; i++) {
        uint32_t tmp = 1 << i;
        if (n == tmp)
            return n;
    }
    assert(false);
    return 0xFFFFFFFF; // for compiler to be happy about returning anything


    // TODO
    // may be better implementation, check it's corectness
// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html
    // return 1 << (32 - _clz(n - 1) + 1); // +1 because it counts from int most significant bit (31th)
}


// TODO: to chyba nie pokazuje najwyższego bitu (od 31)
__device__ 
void binprintf(uint32_t v)
{
    uint32_t mask = 1 << ((sizeof(uint32_t) << 3) - 1);
    while (mask) {
        printf("%u", (v & mask ? 1 : 0));
        mask >>= 1;
    }
    printf("\n");
}

__device__
int getGlobalIdx(){
    return blockIdx.x * blockDim.x * blockDim.y
    + threadIdx.y * blockDim.x + threadIdx.x;
}

__global__ 
void updateSpecific(uint32_t* indices, uint32_t indicesNum, uint32_t* from, uint32_t* to) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid + 1 > indicesNum) {
        return;
    }
    printf("%d: updatuję %d\n", tid, indices[tid]);
    to[indices[tid]] = from[indices[tid]];
}

/**
 * Max number of threadIdx.x is 2^16, sometimes
 * we need more but still use only x indexing, thus need to
 * use several blocks. 
 */
__host__ std::pair<uint16_t, uint16_t> getBlockThreadSplit(uint32_t threads) {
    if (threads < (1 << 16)) {
        return std::make_pair(1, (uint16_t) threads);
    }
    assert(threads < (1 << 26)); // max blockIdx.x * threadIdx.x
    return std::make_pair((uint16_t)  ceil( (float) threads / (float) (1 << 16)), (uint16_t) (1 << 16));
}

__host__
void print_comm_assignment(const uint32_t V_MAX_IDX, const uint32_t* __restrict__ comm) {
    auto v = std::vector<std::set<uint32_t>> (V_MAX_IDX + 1);

    for (int i = 1; i <= V_MAX_IDX; i++) {
        v[comm[i]].insert(i);
    }

    for (int i = 1; i <= V_MAX_IDX; i++) {
        if (v[i].size() == 0)
            continue;
        
        printf("%d ", i);
        std::copy(v[i].begin(), v[i].end(), std::ostream_iterator<uint32_t>(std::cout, " "));
        // PRINT(v[i].begin(), v[i].end());
        printf("\n");
    }
}