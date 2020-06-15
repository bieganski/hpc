#pragma once

#include <fstream>
#include <vector>
#include <mpi.h>
#include <cassert>

#include "types.hpp"

extern bool VERBOSE;
extern char* FILE_PATH_IN;
extern char* FILE_PATH_OUT;
extern int STEP_COUNT;
extern double DELTA_TIME;
extern size_t N;
extern int NUM_PROC;



#define ROOT_NODE 0
#define NULL_TAG 0  // TODO MPI_TAG_ANY

#define NEXT(rank) (rank + 1) % NUM_PROC
#define PREV(rank) (rank == 0 ? NUM_PROC - 1 : rank - 1)

#define MOD3(num) (num % 3 == 0)

int parse_args(int argc, char **argv);

std::ifstream get_input_content();

std::vector<MsgBuf*> parse_input(std::ifstream content);

void handle_redundant_nodes(int myRank);

// inclusive
inline size_t MIN_PART_IDX(int rank) {
    assert(rank <= NUM_PROC);
    int numBase = N / NUM_PROC;
    int numInc = N % NUM_PROC;
    int idx = rank * numBase + (rank > numInc ? numInc : rank);
    return idx;
}

// exclusive
inline size_t MAX_PART_IDX(int rank) {
    return MIN_PART_IDX(rank + 1);
}

void print_msg_buf(MsgBuf* buf);

MsgBuf* distribute_bufs(std::vector<MsgBuf*>& vec, int myRank);

void clear_buffers(std::vector<MsgBuf*>& vec);

void dump_results(char* gatherBuf, size_t dataSize, std::string fileName);

std::vector<MsgBuf*> collect_results(MsgBuf* myDataPtr, char* gatherPtr, size_t dataSize, int rank);

#define BUF_ISEND(__buf, __numBytes, __whom, __reqPtr) \
        do { \
            MPI_Isend((void*) __buf, __numBytes, MPI_BYTE, __whom, 0, ACTIVE_NODES_WORLD, __reqPtr); \
        } \
        while(0)

#define BUF_RECV(__buf, __numBytes, __fromWhom) \
        do { \
            MPI_Recv((void*) __buf, __numBytes, MPI_BYTE, __fromWhom, NULL_TAG, ACTIVE_NODES_WORLD, MPI_STATUS_IGNORE); \
            INIT_BUF(__buf); \
        } \
        while(0)

#define BUF_IRECV(__buf, __numBytes, __fromWhom, __reqPtr) \
        do { \
            MPI_Irecv((void*) __buf, __numBytes, MPI_BYTE, __fromWhom, NULL_TAG, ACTIVE_NODES_WORLD, __reqPtr); \
            INIT_BUF(__buf); \
        } \
        while(0)
