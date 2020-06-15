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

MPI_Comm ACTIVE_NODES_WORLD;

int main(int argc, char **argv) {
    parse_args(argc, argv);

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_PROC);

    int myRank;
    std::vector<MsgBuf*> bufs;

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    MsgBuf *myBuf;
    if (myRank == ROOT_NODE) {
        bufs = parse_input(get_input_content());
        MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, ROOT_NODE, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, ROOT_NODE, MPI_COMM_WORLD);
        assert(N > 0);
        assert(DELTA_TIME > 0.0);
    }

    // Some nodes may not be used, if there is not enough data for everyone.
    // At the same time, we want to use collectives (i.a. MPI_Gather), thus create new world
    int color = compute_color(myRank);
    int res = MPI_Comm_split(MPI_COMM_WORLD, color, myRank, &ACTIVE_NODES_WORLD);
    assert(res == MPI_SUCCESS);
    handle_redundant_nodes(myRank);

    myBuf = distribute_bufs(bufs, myRank);
    assert(bufs.empty());

    // here all the nodes got their particles subset in memory.
    // first algorithm run updates only acceleration ('first_iter' true)
    body_algo(myRank, myBuf, true);

    size_t dataSize = MAX_BUF_SIZE * NUM_PROC;
    char *gatherBuf = myRank == ROOT_NODE ? (char*) calloc(dataSize, 1) : NULL;

    bufs = collect_results(myBuf, gatherBuf, dataSize, myRank);
    
    
    for (int i = 0; i < STEP_COUNT; i++) {
        myBuf = distribute_bufs(bufs, myRank);
        body_algo(myRank, myBuf, false);
        
        // after all phase, each thread's result is in it's 'myBuf', need to gather them
        bufs = collect_results(myBuf, gatherBuf, dataSize, myRank);

        if (myRank == ROOT_NODE && VERBOSE) {
            std::string out(FILE_PATH_OUT);
            out.append("_" + std::to_string(i + 1) + ".txt");
            dump_results(gatherBuf, dataSize, out);
        }
    }

    if (myRank == ROOT_NODE) {
        std::string out(FILE_PATH_OUT);
        out.append("_stepcount.txt");
        dump_results(gatherBuf, dataSize, out);
    }
    
    MPI_Finalize();

    return 0;
}