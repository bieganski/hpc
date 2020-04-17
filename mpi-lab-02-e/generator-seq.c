/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include "graph-base.h"
#include "graph-utils.h"



int main(int argc, char * argv[])
{
  int numVertices = 0;
  my_graph_part_t * graph;

  MPI_Init(&argc, &argv);
  
#ifdef USE_RANDOM_GRAPH
#ifdef USE_RANDOM_SEED
  srand(USE_RANDOM_SEED);
#endif
#endif

  if (argc == 2)
  {
    numVertices = atoi(argv[1]);
  }
  
  if (numVertices <= 0)
  {
    fprintf(stderr, "Usage: %s <num_vertices>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }
  
  graph = myCreateAndDistributeGraph(numVertices, 1 /* numProcesses */, 0 /* myRank */);
  if (graph == NULL)
  {
    fprintf(stderr, "Error creating the graph.\n");
    MPI_Finalize();
    return 2;
  }
  
  myCollectAndPrintGraph(graph, 1 /* numProcesses */, 0 /* myRank */);
  
  myDestroyGraph(graph, 1 /* numProcesses */, 0 /* myRank */);
  
  MPI_Finalize();

  return 0;
}
