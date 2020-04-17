/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "graph-base.h"



my_graph_part_t * myAllocGraphPart(
    int numVertices,
    int firstRowIdxIncl,
    int lastRowIdxExcl
)
{
  my_graph_part_t *   g;
  int            i, n;

  if (firstRowIdxIncl >= lastRowIdxExcl ||
      firstRowIdxIncl < 0 || numVertices <= 0)
  {
    return NULL;
  }
  g = (my_graph_part_t *)malloc(sizeof(my_graph_part_t));
  if (g == NULL)
  {
    return NULL;
  }
  g->data = NULL;
  g->extraRow = NULL;
  g->numVertices = numVertices;
  g->firstRowIdxIncl = firstRowIdxIncl;
  g->lastRowIdxExcl = lastRowIdxExcl;
  g->data =
      (int * *)malloc(
          sizeof(int *) * (g->lastRowIdxExcl - g->firstRowIdxIncl)
      );
  if (g->data == NULL)
  {
    myFreeGraphPart(g);
    return NULL;
  }
  g->extraRow = (int *)malloc(sizeof(int) * g->numVertices);
  if (g->extraRow == NULL)
  {
    myFreeGraphPart(g);
    return NULL;
  }
  n = g->lastRowIdxExcl - g->firstRowIdxIncl;
  for (i = 0; i < n; ++i)
  {
    g->data[i] = NULL;
  }
  for (i = 0; i < n; ++i)
  {
    g->data[i] = (int *)malloc(sizeof(int) * g->numVertices);
    if (g->data[i] == NULL)
    {
      myFreeGraphPart(g);
      return NULL;
    }
  }
  return g;
}



void myInitGraphRow(
    int * row,
    int rowIdx,
    int numVertices
)
{
  int j;
  for (j = 0; j < numVertices; ++j)
  {
    row[j] = rowIdx == j ? 0 :
#ifndef USE_RANDOM_GRAPH
          (rowIdx - j == 1 || j - rowIdx == 1 ?
                  1 : numVertices + 5);
#else
          (rand() & 8191) + 1;
#endif
  }
}



void myPrintGraphRow(
    int const * row,
    int rowIdx,
    int numVertices
)
{
  int j;
  printf("%4d", row[0]);
  for (j = 1; j < numVertices; ++j)
  {
    printf(" %4d", row[j]);
  }
  printf("\n");
}



void myFreeGraphPart(
    my_graph_part_t * g
)
{
  if (g == NULL)
  {
    return;
  }
  if (g->extraRow != NULL)
  {
    free(g->extraRow);
    g->extraRow = NULL;
  }
  if (g->data != NULL)
  {
    int i, n;
    for (i = 0, n = g->lastRowIdxExcl - g->firstRowIdxIncl; i < n; ++i)
    {
      if (g->data[i] != NULL)
      {
        free(g->data[i]);
        g->data[i] = NULL;
      }
    }
    free(g->data);
    g->data = NULL;
  }
  g->numVertices = 0;
  g->firstRowIdxIncl = 0;
  g->lastRowIdxExcl = 0;
}
