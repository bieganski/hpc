#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <float.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <iomanip>

#include <mpi.h>

#include "utils.hpp"
#include "types.hpp"

extern MPI_Comm ACTIVE_NODES_WORLD;

int parse_args(int argc, char **argv) {

    opterr = 0;

    if (argc != 5 && argc != 6) {
        fprintf (stderr, "Usage: ./body3 particles_in.txt particles_out stepcount deltatime [-v]\n");
        exit(1);
    }
    FILE_PATH_IN = argv[1];
    FILE_PATH_OUT = argv[2];
    try {
        STEP_COUNT = std::stoi(std::string(argv[3]));
    } catch  (... ) {
        fprintf (stderr, "step_count must be proper unsigned int!\n");
        exit(1);
    }
    try {
        DELTA_TIME = std::stod(std::string(argv[4]));
    } catch  (... ) {
        fprintf (stderr, "delta_time must be proper double!\n");
        exit(1);
    }

    if (argc == 6) {
        if (std::strcmp(argv[5], "-v")) {
            fprintf (stderr, "last argument expected is '-v'!\n");
            exit(1);
        }
        VERBOSE = true;
    } else {
        VERBOSE = false;
    }

    return 0;
}


std::ifstream get_input_content() {
    try {
        std::ifstream file_stream{FILE_PATH_IN};
        return file_stream;
    } catch (const char * f) {
        std::cerr << "Error reading input file" + std::string(FILE_PATH_IN);
        exit(1);
    }
}

void handle_redundant_nodes(int myRank) {
    if (NUM_PROC > N) {
        NUM_PROC = N;
    }
    if (myRank >= NUM_PROC || ((MOD3(NUM_PROC) && myRank == NUM_PROC - 1))) {
        // i'm redundant
        MPI_Finalize();
        exit(0);
    }
    if (MOD3(NUM_PROC)) {
        NUM_PROC -= 1;
    }
}

int compute_color(int myRank) {
    if (N <= NUM_PROC)
        return 123;
    int res = myRank + 1 > N ? MPI_UNDEFINED : 123;
    if (MOD3(N) && myRank == N - 1) {
        return MPI_UNDEFINED;
    }
    return res;
}

std::vector<MsgBuf*> parse_input(std::ifstream content) {
    std::stringstream ss;
    ss << content.rdbuf();
    content.close();
    std::string line;
    std::stringstream tmp_ss;

    std::vector<ParticleDescr> particlesVec;

    while(std::getline(ss, line, '\n')){

        if (line.size() == 0)
            continue;

        tmp_ss = std::stringstream();
        tmp_ss << line;
        
        Pos p;
        Vel v;
        Acc a {.ax = 0.0, .ay = 0.0, .az = 0.0};
        Force f {.fx = 0.0, .fy = 0.0, .fz = 0.0};

        tmp_ss >> p.x;
        tmp_ss >> p.y;
        tmp_ss >> p.z;

        tmp_ss >> v.vx;
        tmp_ss >> v.vy;
        tmp_ss >> v.vz;

        ParticleDescr descr {.pos = p, .vel = v, .acc = a, .force = f};

        particlesVec.push_back(descr);
    }

    N = particlesVec.size();

    std::vector<MsgBuf*> res;

    int numProc = MOD3(NUM_PROC) ? NUM_PROC - 1 : NUM_PROC;

    for (int i = 0; i < numProc; i++) {
        size_t minIdxIncl = MIN_PART_IDX(i), maxIdxExcl = MAX_PART_IDX(i);

        size_t particlesNum = maxIdxExcl - minIdxIncl;

        if (particlesNum == 0) {
            return res;
        }

        MsgBuf *buf = (MsgBuf*) calloc(sizeof(MsgBuf) + sizeof(ParticleDescr) * particlesNum, 1); // zeroed memory

        buf->owner = i;
        buf->particlesNum = particlesNum;

        INIT_BUF(buf);
        
        for (int j = 0; j < particlesNum; j++) {
            buf->elems[j] = particlesVec[minIdxIncl + j];
        }

        res.push_back(buf);
    }
    return res;
}


void print_msg_buf(MsgBuf* buf) {
    printf("MSGBUF: owner: %d, size: %d\n", buf->owner, buf->particlesNum);
    printf("\tfirst: || ");
    PRINT_POS(buf->elems[0].pos);
    printf("\tlast: || ");
    PRINT_POS(buf->elems[buf->particlesNum - 1].pos);
    printf("\tACC: || (%2.10f, %2.10f, %2.10f)\n", ACCX(buf, 0), ACCY(buf, 0), ACCZ(buf, 0));
    printf("\tFFF: || (%2.10f, %2.10f, %2.10f)\n", FX(buf, 0), FY(buf, 0), FZ(buf, 0));
}

// only called by rank 0 node
void free_buffers(std::vector<MsgBuf*>& vec) {
    for (int i = 1; i < vec.size(); ++i) {
        free(vec[i]);
    }
}

/**
 * Process 0 is distributing, doesn't send to itself.
 */
MsgBuf* distribute_bufs(std::vector<MsgBuf*>& vec, int myRank) {
    MsgBuf* res;
    if (myRank == ROOT_NODE) {
        
        MPI_Request *requests = (MPI_Request *) calloc(vec.size(), sizeof(MPI_Request));
        for (int i = 1; i < vec.size(); ++i) {
            int rank = vec[i]->owner;
            assert(BUF_SIZE_RANK(rank) == BUF_SIZE(vec[i]));
            BUF_ISEND(vec[i], BUF_SIZE_RANK(rank), rank, &requests[i]);
        }
        MPI_Waitall(vec.size() - 1, &requests[1], MPI_STATUSES_IGNORE);
        free(requests);
        free_buffers(vec);
        res = vec[0];
        INIT_BUF(res);
        vec.clear();
    } else {
        res = (MsgBuf*) malloc(MAX_BUF_SIZE);
        BUF_RECV(res, BUF_SIZE_RANK(myRank), 0);
        INIT_BUF(res);
        assert(res->owner == myRank);
    }
    return res;
}

// TODO data cleaning
std::vector<MsgBuf*> collect_results(MsgBuf* myDataPtr, char* gatherPtr, size_t dataSize, int rank) {
    std::vector<MsgBuf*> res;
    
    assert(myDataPtr != NULL);
    if (rank == ROOT_NODE) {
        assert(gatherPtr != NULL);
    } else {
        assert(gatherPtr == NULL);
    }

    MPI_Gather(myDataPtr, MAX_BUF_SIZE, MPI_CHAR, gatherPtr, MAX_BUF_SIZE, MPI_CHAR, ROOT_NODE, ACTIVE_NODES_WORLD);

    if (rank != ROOT_NODE)
        return res;

    for (int i = 0; i < dataSize; i+= MAX_BUF_SIZE) {
        MsgBuf *out, *tmp = (MsgBuf*) (gatherPtr + i);
        out = (MsgBuf *) calloc(MAX_BUF_SIZE, 1);
        memcpy(out, tmp, MAX_BUF_SIZE);
        INIT_BUF(out);
        res.push_back(out);
    }

    return res;
}


void dump_results(char* gatherBuf, size_t dataSize, std::string fileName) {
    std::ofstream out(fileName);
    out << std::fixed << std::setprecision(16);
    for (int i = 0; i < dataSize; i += MAX_BUF_SIZE) {
        MsgBuf *tmp = (MsgBuf *) (gatherBuf + i);
        INIT_BUF(tmp);
        for (int j = 0; j < tmp->particlesNum; j++) {
            out << POSX(tmp, j) << " " << POSY(tmp, j) << " " << POSZ(tmp, j) << " ";
            out << VELX(tmp, j) << " " << VELY(tmp, j) << " " << VELZ(tmp, j) << "\n";
        }
    }
}