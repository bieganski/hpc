#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <float.h>

#include <mpi.h>

#include <iostream>

#include "utils.hpp"
#include "body.hpp"

bool VERBOSE;
char* FILE_PATH_IN;
char* FILE_PATH_OUT;
int STEP_COUNT;
double DELTA_TIME;

size_t N;
int NUM_PROC;

const double EPS = DBL_EPSILON;

void handle_redundant_nodes(int myRank) {
    if (myRank >= N) {
        // i'm redundant
        printf("### IM REDUNDANT: %d\n", myRank);
        MPI_Finalize();
        exit(0);
    } else if (NUM_PROC > N) {
        printf(">> REDUNDANT: obcinam %d do %d\n", NUM_PROC, N);
        NUM_PROC = N;
    }
}

int main(int argc, char **argv) {
    parse_args(argc, argv);

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_PROC);

    int myRank;

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    MsgBuf *myBuf;
    if (myRank == 0) {
        auto bufs = parse_input(get_input_content());

        MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); // broadcast value of N

        handle_redundant_nodes(myRank);

        mpi_distribute(bufs); // TODO - now it may send uneccessary 16 bytes chunks to killed procs.
        myBuf = bufs[0];
        clear_buffers(bufs);
    } else {
        MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        assert(N > 0);

        handle_redundant_nodes(myRank);

        myBuf = (MsgBuf*) malloc(MAX_BUF_SIZE);
        BUF_RECV(myBuf, BUF_SIZE_RANK(myRank), 0);
        assert(myBuf->owner == myRank);
    }

    // here all the nodes got their particles subset in memory.
    body_algo(myRank, myBuf);

    MPI_Finalize();

    return 0;
}