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

#include <mpi.h>

#include "utils.hpp"
#include "types.hpp"

int parse_args(int argc, char **argv) {
    int gflag = 0, fflag = 0;
    int c;

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

        tmp_ss >> p.x;
        tmp_ss >> p.y;
        tmp_ss >> p.z;

        tmp_ss >> v.vx;
        tmp_ss >> v.vy;
        tmp_ss >> v.vz;

        PRINT_POS(p);
        PRINT_VEL(v);

        ParticleDescr descr {.pos = p, .vel = v, .acc = a};

        particlesVec.push_back(descr);
    }

    N = particlesVec.size();

    std::vector<MsgBuf*> res;

    for (int i = 0; i < NUM_PROC; i++) {
        size_t minIdxIncl = MIN_PART_IDX(i), maxIdxExcl = MAX_PART_IDX(i);

        size_t particlesNum = maxIdxExcl - minIdxIncl;

        if (particlesNum == 0) {
            printf("early return because of proccess flow\n");
            return res;
        }

        printf("%d) partnum: %d\n", i, particlesNum);

        std::cerr << "malloc of size " << sizeof(MsgBuf) + sizeof(ParticleDescr) * particlesNum << "\n";
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
}

/**
 * Process 0 is distributing, doesn't send to itself.
 */
void mpi_distribute(std::vector<MsgBuf*>& vec) {
    MPI_Request requests[vec.size()];
    for (int i = 1; i < vec.size(); ++i) {
        int rank = vec[i]->owner;
        assert(BUF_SIZE_RANK(rank) == BUF_SIZE(vec[i]));
        BUF_ISEND(vec[i], BUF_SIZE_RANK(rank), rank, &requests[i]);
    }
    MPI_Waitall(vec.size() - 1, &requests[1], MPI_STATUSES_IGNORE);
}

// only called by rank 0 node
void clear_buffers(std::vector<MsgBuf*>& vec) {
    for (int i = 1; i < vec.size(); ++i) {
        free(vec[i]);
    }
}

