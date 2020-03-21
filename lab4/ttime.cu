#include <time.h>
#include <stdio.h>

#include "ttime.h"

static cudaEvent_t start, stop;

static cudaEvent_t start2, stop2;

static struct timespec cpu_start, cpu_stop;

static bool second = false;

void start_time_cuda() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    if (second)
        cudaEventRecord( start2, 0 );
    else
        cudaEventRecord( start, 0 );
}


void start_time_cpu() {
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
}

void stop_time_cuda() {
    float elapsedTime;
    if (second) {
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        cudaEventElapsedTime( &elapsedTime, start2, stop2);  
        printf("SECOND: Total GPU execution time:  %3.4f ms\n", elapsedTime);  
    } else {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &elapsedTime, start, stop);
        printf("FIRST: Total GPU execution time:  %3.4f ms\n", elapsedTime);  
    }
    second = true;
}


void stop_time_cpu() {
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
  double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf( "CPU execution time:  %3.4f ms\n", result);
}