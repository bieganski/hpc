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
#define NULL_TAG 0

#define NEXT(rank) (rank + 1) % NUM_PROC
#define PREV(rank) (rank == 0 ? NUM_PROC - 1 : rank - 1)

int parse_args(int argc, char **argv);

std::ifstream get_input_content();

std::vector<MsgBuf*> parse_input(std::ifstream content);

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

void mpi_distribute(std::vector<MsgBuf*>& vec);

void clear_buffers(std::vector<MsgBuf*>& vec);

#define BUF_ISEND(__buf, __numBytes, __whom, __reqPtr) \
        do { \
            printf("Sending %d bytes to %d\n", __numBytes, __whom); \
            MPI_Isend((void*) __buf, __numBytes, MPI_BYTE, __whom, 0, MPI_COMM_WORLD, __reqPtr); \
        } \
        while(0)

#define BUF_RECV(__buf, __numBytes, __fromWhom) \
        do { \
            printf("Receiving %d bytes from %d\n", __numBytes, __fromWhom); \
            MPI_Recv((void*) __buf, __numBytes, MPI_BYTE, __fromWhom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); \
            INIT_BUF(__buf); \
            printf("Odebrałem: "); \
            print_msg_buf(__buf); \
        } \
        while(0)
