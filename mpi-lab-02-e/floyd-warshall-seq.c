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



static void floydWarshallSeq(my_graph_part_t * g)
{
  int k, i, j, m;
  /* For the sequential version, we assume the entire graph. */
  assert(g->firstRowIdxIncl == 0 && g->lastRowIdxExcl == g->numVertices);
  m = g->numVertices;
  for (k = 0; k < m; ++k)
  {
    for (i = 0; i < m; ++i)
    {
      for (j = 0; j < m; ++j)
      {
        int pathSum = g->data[i][k] + g->data[k][j];
        if (g->data[i][j] > pathSum)
        {
          g->data[i][j] = pathSum;
        }
      }
    }
  }
}



int main(int argc, char * argv[])
{
  int numVertices = 0;
  int showResults = 0;
  int i;
  my_graph_part_t * graph;
  double startTime;
  double endTime;

  MPI_Init(&argc, &argv);
  
#ifdef USE_RANDOM_GRAPH
#ifdef USE_RANDOM_SEED
  srand(USE_RANDOM_SEED);
#endif
#endif

  for (i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], "--show-results") == 0)
    {
      showResults = 1;
    }
    else
    {
      numVertices = atoi(argv[i]);
    }
  }
  
  if (numVertices <= 0)
  {
    fprintf(stderr, "Usage: %s [--show-results] <num_vertices>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }
  
  fprintf(stderr, "Running the Floyd-Warshall algorithm for a graph with %d vertices.\n", numVertices);
  
  graph = myCreateAndDistributeGraph(numVertices, 1 /* numProcesses */, 0 /* myRank */);
  if (graph == NULL)
  {
    fprintf(stderr, "Error distributing the graph for the algorithm.\n");
    MPI_Finalize();
    return 2;
  }

  if (showResults)
  {
    myCollectAndPrintGraph(graph, 1 /* numProcesses */, 0 /* myRank */);
  }
  
  startTime = MPI_Wtime();
  
  floydWarshallSeq(graph);
  
  endTime = MPI_Wtime();
  
  fprintf(
      stderr,
      "The time required for the Floyd-Warshall algorithm on a %d-node graph with %d process(es): %f.\n",
      numVertices,
      1, /* numProcesses */
      endTime - startTime
  );
  
  if (showResults)
  {
    myCollectAndPrintGraph(graph, 1 /* numProcesses */, 0 /* myRank */);
  }
  
  myDestroyGraph(graph, 1 /* numProcesses */, 0 /* myRank */);
  
  MPI_Finalize();

  return 0;
}
