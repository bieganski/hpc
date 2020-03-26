#include "./common/helpers.h"

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x + blockDim.x;
    if (tid < N) {
        int tid1 = (tid + 1) % 256;
        int tid2 = (tid + 2) % 256;
        float aSum = (a[tid] + a[tid1] + a[tid2]) / 3.0f;
        float bSum = (b[tid] + b[tid1] + b[tid2]) / 3.0f;
        c[tid] = (aSum + bSum) / 2;
    }
}

int main(void) {
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));


    // TODO
    // uncomment for measuring time using single, defualt null stream 
    // for (int i = 0; i < FULL_DATA_SIZE; i += N) {
    //     HANDLE_ERROR(cudaMemcpy(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice));
    //     HANDLE_ERROR(cudaMemcpy(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice));

    //     kernel<<<N / 256, 256>>>(dev_a, dev_b, dev_c);

    //     HANDLE_ERROR(cudaMemcpy(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    // }
    int parity = 0;

    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        parity = 1 - parity;
        
        if (parity) {
            HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
            HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
            kernel<<<N / 256, 256, 0, stream1>>>(dev_a, dev_b, dev_c);
            HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
        } else {
            HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream2));
            HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream2));
            kernel<<<N / 256, 256, 0, stream2>>>(dev_a, dev_b, dev_c);
            HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream2));
        }
    }

    HANDLE_ERROR(cudaStreamSynchronize(stream1));
    HANDLE_ERROR(cudaStreamSynchronize(stream2));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time taken: %3.1f ms\n", elapsedTime);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    
    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    HANDLE_ERROR(cudaStreamDestroy(stream1));
    HANDLE_ERROR(cudaStreamDestroy(stream2));
    
    return 0;
}
