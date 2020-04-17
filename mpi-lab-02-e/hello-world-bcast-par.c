/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#define MSG_PREFIX "Hello world from process "
#define MSG_FORMAT "%4d"
#define MSG_SUFFIX "!"

int main(int argc, char * argv[])
{
  int myProcessNo;
  int numProcesses;
  char message[256];
  int srcProcessNo = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myProcessNo);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
   
  if (myProcessNo == 0)
  {
    /* Proces wysylajacy komunikat "Hello world..." */
    int messageLen = snprintf(
        message,
        sizeof(message) / sizeof(char),
        MSG_PREFIX MSG_FORMAT MSG_SUFFIX,
        myProcessNo
    ) + 1;
    assert(messageLen <= sizeof(message) / sizeof(char));
    assert(messageLen == strlen(MSG_PREFIX) + 4 + strlen(MSG_SUFFIX) + 1);
    MPI_Bcast(
        message,
        messageLen,
        MPI_CHAR,
        srcProcessNo,
        MPI_COMM_WORLD
    );
  }
  else
  {
    /* Proces odbierajacy komunikaty "Hello world..." */
    MPI_Bcast(
        message,
        strlen(MSG_PREFIX) + 4 + strlen(MSG_SUFFIX) + 1,
        MPI_CHAR,
        srcProcessNo,
        MPI_COMM_WORLD
    );
    printf("Process %d received broadcast message \"%s\"\n", myProcessNo, message);
  }

  MPI_Finalize();

  return 0;
}
