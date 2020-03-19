#include <time.h>
#include <stdio.h>

#include "ttime.h"

static cudaEvent_t start, stop;

static struct timespec cpu_start, cpu_stop;

bool inited = false;

void start_time_cuda() {
    if (!inited) {
        inited = true;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    cudaEventRecord( start, 0 );
}


void start_time_cpu() {
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
}

void stop_time_cuda() {
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop);
  printf("Total GPU execution time:  %3.1f ms\n", elapsedTime);
}


void stop_time_cpu() {
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
  double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf( "CPU execution time:  %3.1f ms\n", result);
}