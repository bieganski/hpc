/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "graph-base.h"
#include "graph-utils.h"



my_graph_part_t * myCreateAndDistributeGraph(
    int numVertices,
    int numProcesses,
    int myRank
)
{
  my_graph_part_t *   g;
  int                 i, n;
  assert(numProcesses == 1 && myRank == 0);
  g = myAllocGraphPart(numVertices, 0, numVertices);
  if (g == NULL)
  {
    return NULL;
  }
  assert(g->numVertices > 0 && g->numVertices == numVertices);
  assert(g->firstRowIdxIncl == 0 && g->lastRowIdxExcl == g->numVertices);
  n = g->numVertices;
  for (i = 0; i < n; ++i)
  {
    myInitGraphRow(g->data[i], i, g->numVertices);
  }
  return g;
}



void myCollectAndPrintGraph(
    my_graph_part_t * g,
    int numProcesses,
    int myRank    
)
{
  int   i, n;
  assert(numProcesses == 1 && myRank == 0);
  assert(g->numVertices > 0);
  assert(g->firstRowIdxIncl == 0 && g->lastRowIdxExcl == g->numVertices);
  n = g->numVertices;
  for (i = 0; i < n; ++i)
  {
    myPrintGraphRow(g->data[i], i, g->numVertices);
  }
}



void myDestroyGraph(
    my_graph_part_t * g,
    int numProcesses,
    int myRank    
)
{
  myFreeGraphPart(g);
}
