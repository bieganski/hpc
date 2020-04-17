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



static inline int myFirstGraphRowOfProcess(
    int numVertices,
    int numProcesses,
    int myRank
)
{
  /* FIXME: implement */
  return myRank;
}



my_graph_part_t * myCreateAndDistributeGraph(
    int numVertices,
    int numProcesses,
    int myRank
)
{
  my_graph_part_t *   g;
  assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
  g =
      myAllocGraphPart(
          numVertices,
          myFirstGraphRowOfProcess(numVertices, numProcesses, myRank),
          myFirstGraphRowOfProcess(numVertices, numProcesses, myRank + 1)
      );
  if (g == NULL)
  {
    return NULL;
  }
  assert(g->numVertices > 0 && g->numVertices == numVertices);
  assert(g->firstRowIdxIncl >= 0 && g->lastRowIdxExcl <= g->numVertices);

  /* FIXME: implement */

  return g;
}



void myCollectAndPrintGraph(
    my_graph_part_t * g,
    int numProcesses,
    int myRank    
)
{
  assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
  assert(g->numVertices > 0);
  assert(g->firstRowIdxIncl >= 0 && g->lastRowIdxExcl <= g->numVertices);

  /* FIXME: implement */

}



void myDestroyGraph(
    my_graph_part_t * g,
    int numProcesses,
    int myRank    
)
{
  myFreeGraphPart(g);
}
