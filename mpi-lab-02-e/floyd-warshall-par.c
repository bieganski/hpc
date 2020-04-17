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



static void floydWarshallPar(my_graph_part_t * g, int numProcesses, int myRank)
{
  assert(numProcesses <= g->numVertices);

  /* FIXME: implement */

}



int main(int argc, char * argv[])
{
  int numVertices = 0;
  int numProcesses = 0;
  int myRank = 0;
  int showResults = 0;
  int i;
  my_graph_part_t * graph;
  double startTime;
  double endTime;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  
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

  if (numProcesses > numVertices)
  {
    numProcesses = numVertices;
    if (myRank >= numProcesses)
    {
      MPI_Finalize();
      return 0;
    }
  }
  
  fprintf(stderr, "Running the Floyd-Warshall algorithm for a graph with %d vertices.\n", numVertices);
  
  graph = myCreateAndDistributeGraph(numVertices, numProcesses, myRank);
  if (graph == NULL)
  {
    fprintf(stderr, "Error distributing the graph for the algorithm.\n");
    MPI_Finalize();
    return 2;
  }

  if (showResults)
  {
    myCollectAndPrintGraph(graph, numProcesses, myRank);
  }
  
  startTime = MPI_Wtime();
  
  floydWarshallPar(graph, numProcesses, myRank);
  
  endTime = MPI_Wtime();
  
  fprintf(
      stderr,
      "The time required for the Floyd-Warshall algorithm on a %d-node graph with %d process(es): %f.\n",
      numVertices,
      numProcesses,
      endTime - startTime
  );
  
  if (showResults)
  {
    myCollectAndPrintGraph(graph, numProcesses, myRank);
  }
  
  myDestroyGraph(graph, numProcesses, myRank);
  
  MPI_Finalize();

  return 0;
}
