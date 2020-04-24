/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */
#include <cstring>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <stdio.h>

#include <mpi.h>
#include "graph-base.h"
#include "graph-utils.h"

int getFirstGraphRowOfProcess(int numVertices, int numProcesses, int myRank) {
    int chunkSize = numVertices % numProcesses ? 
        (numVertices / numProcesses) + 1 
        : (numVertices / numProcesses);
    
    return std::min(myRank * chunkSize, numVertices);
}

Graph* createAndDistributeGraph(int numVertices, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);

    auto myGraph = allocateGraphPart(
            numVertices,
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank),
            getFirstGraphRowOfProcess(numVertices, numProcesses, myRank + 1)
    );

    if (myGraph == nullptr) {
        return nullptr;
    }

    assert(myGraph->numVertices > 0 && myGraph->numVertices == numVertices);
    assert(myGraph->firstRowIdxIncl >= 0 && myGraph->lastRowIdxExcl <= myGraph->numVertices);


    int rowMemSize = numVertices * sizeof(int);
    int myRows = myGraph->lastRowIdxExcl - myGraph->firstRowIdxIncl;
    int* buf = new int[myRows * numVertices]; // after while of thinking you can see it will be enough

    if (myRank > 0) {
        MPI_Recv(buf, myRows * numVertices, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int r  = 0; r < myRows; r++) {
            std::memcpy(myGraph->data[r], buf + (r * numVertices), rowMemSize);
        }

    } else {
        assert(myRank == 0);

        // initialize my graph

        for (int i = 0; i < myRows; ++i) {
            initializeGraphRow(myGraph->data[i], i, numVertices);
        }

        for (int p = 1; p < numProcesses; p++) {

            auto onesGraph = allocateGraphPart(
                numVertices,
                getFirstGraphRowOfProcess(numVertices, numProcesses, p),
                getFirstGraphRowOfProcess(numVertices, numProcesses, p + 1)
            );

            int onesRows = onesGraph->lastRowIdxExcl - onesGraph->firstRowIdxIncl;

            for (int i = 0; i < onesRows; ++i) {
                initializeGraphRow(&buf[i * numVertices], onesGraph->firstRowIdxIncl + i, numVertices);
            }

            MPI_Send(buf, onesRows * numVertices, MPI_INT, p, 0, MPI_COMM_WORLD);
        }

    }

    delete []buf;

    return myGraph;
}


void collectAndPrintGraph(Graph* graph, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
    assert(graph->numVertices > 0);
    assert(graph->firstRowIdxIncl >= 0 && graph->lastRowIdxExcl <= graph->numVertices);

    /* FIXME: implement */

    int myRows = graph->lastRowIdxExcl - graph->firstRowIdxIncl;

    int numVertices = graph->numVertices;
    int *buf = new int[numVertices * myRows]; // it is enough, even for 0 rank node

    int rowMemSize = numVertices * sizeof(int);

    if (myRank > 0) {
        for (int r = 0; r < myRows; r++) {
            std::memcpy(buf + r * numVertices, graph->data[r], rowMemSize);
        }
        MPI_Send(buf, myRows * numVertices, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        int currIdx = graph->lastRowIdxExcl;

        auto wholeGraph = allocateGraphPart(numVertices, 0, numVertices);

        
        for (int p = 1; p < numProcesses; p++) {
            int onesRows = getFirstGraphRowOfProcess(numVertices, numProcesses, p + 1) - getFirstGraphRowOfProcess(numVertices, numProcesses, p);

            MPI_Recv(buf, onesRows * numVertices, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = currIdx; i < currIdx + onesRows; i++) {
                std::memcpy(wholeGraph->data[i], buf + (i - currIdx) * numVertices, rowMemSize);
            }

            currIdx += onesRows;
        }

        // and at the end add my graph part
        for (int i = 0; i < myRows; i++) {
            std::memcpy(wholeGraph->data[i], graph->data[i], rowMemSize);
        }


        for (int i = 0; i < wholeGraph->numVertices; i++) {
            for (int j = 0; j < wholeGraph->numVertices; j++) {
                std::cout << wholeGraph->data[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void destroyGraph(Graph* graph, int numProcesses, int myRank) {
    freeGraphPart(graph);
}
