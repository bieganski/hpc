#include "cuda.h"
#include "common/errors.h"
#include "common/cpu_bitmap.h"

// #include <helper_cuda.h>

#include "ttime.h"

#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define SPHERES 1000

struct Sphere {
	float red, green, blue;
	float radius;
	float x, y, z;

	__device__ float hit(float bitmapX, float bitmapY, float *colorFalloff) {
		float distX = bitmapX - x;
		float distY = bitmapY - y;

		if (distX * distX + distY * distY < radius * radius) { 
			float distZ = sqrtf(radius * radius - distX * distX - distY * distY);
			*colorFalloff = distZ / sqrtf(radius * radius);
			return distZ + z; 
		}

		return -INF;
	}
};

__global__ void kernel(Sphere *spheres, unsigned char* bitmap) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float bitmapX = (x - DIM / 2);
	float bitmapY = (y - DIM / 2);
	
	float red = 0, green = 0, blue = 0;
	float maxDepth = -INF;
	
	for (int _i = 0; _i < 1000 * SPHERES; _i++) {
		int i = _i % (SPHERES + 1);
		float colorFalloff;
		float depth = spheres[i].hit(bitmapX, bitmapY, &colorFalloff);
		
		if (depth > maxDepth) { 
			red = spheres[i].red * colorFalloff;
			green = spheres[i].green * colorFalloff;
			blue = spheres[i].blue * colorFalloff;
			maxDepth = depth; 
		}
	}

	bitmap[offset * 4 + 0] = (int) (red * 255);
	bitmap[offset * 4 + 1] = (int) (green * 255);
	bitmap[offset * 4 + 2] = (int) (blue * 255);
	bitmap[offset * 4 + 3] = 255;
}

__constant__ Sphere spheres[SPHERES];

__global__ void kernel2(unsigned char* bitmap) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float bitmapX = (x - DIM / 2);
	float bitmapY = (y - DIM / 2);
	
	float red = 0, green = 0, blue = 0;
	float maxDepth = -INF;
	
	for (int i = 0; i < SPHERES; i++) { 
		float colorFalloff;
		float depth = spheres[i].hit(bitmapX, bitmapY, &colorFalloff);
		
		if (depth > maxDepth) { 
			red = spheres[i].red * colorFalloff;
			green = spheres[i].green * colorFalloff;
			blue = spheres[i].blue * colorFalloff;
			maxDepth = depth; 
		}
	}

	bitmap[offset * 4 + 0] = (int) (red * 255);
	bitmap[offset * 4 + 1] = (int) (green * 255);
	bitmap[offset * 4 + 2] = (int) (blue * 255);
	bitmap[offset * 4 + 3] = 255;
}


texture<float, 1> r;
texture<float, 1> g;
texture<float, 1> b;

__global__ void kernel3(Sphere *spheres, unsigned char* bitmap) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float bitmapX = (x - DIM / 2);
	float bitmapY = (y - DIM / 2);
	
	float red = 0, green = 0, blue = 0;
	float maxDepth = -INF;
	
	for (int _i = 0; _i < 1000 * SPHERES; _i++) {
		int i = _i % (SPHERES + 1);
		float colorFalloff;
		float depth = spheres[i].hit(bitmapX, bitmapY, &colorFalloff);
		
		if (depth > maxDepth) {
			red = tex1Dfetch(r, i) * colorFalloff;
			green = tex1Dfetch(g, i) * colorFalloff;
			blue = tex1Dfetch(b, i) * colorFalloff;
			maxDepth = depth; 
		}
	}

	bitmap[offset * 4 + 0] = (int) (red * 255);
	bitmap[offset * 4 + 1] = (int) (green * 255);
	bitmap[offset * 4 + 2] = (int) (blue * 255);
	bitmap[offset * 4 + 3] = 255;
}

struct DataBlock {
	unsigned char *hostBitmap;
	Sphere *spheres;
};

int main(void) {
	DataBlock data;
	
	CPUBitmap bitmap(DIM, DIM, &data);
	unsigned char *devBitmap;
	Sphere *devSpheres;


	Sphere *hostSpheres = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	
	for (int i = 0; i < SPHERES; i++) {
		hostSpheres[i].red = rnd(1.0f);
		hostSpheres[i].green = rnd(1.0f);
		hostSpheres[i].blue = rnd(1.0f);
		hostSpheres[i].x = rnd(1000.0f) - 500;
		hostSpheres[i].y = rnd(1000.0f) - 500;
		hostSpheres[i].z = rnd(1000.0f) - 500;
		hostSpheres[i].radius = rnd(100.0f) + 20;
	}
	
	HANDLE_ERROR(cudaMalloc((void**)&devBitmap, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**)&devSpheres, sizeof(Sphere) * SPHERES));


	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);



	HANDLE_ERROR(cudaMemcpy(devSpheres, hostSpheres, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));

	start_time_cuda();

	kernel<<<grids,threads>>>(devSpheres, devBitmap);
	
	stop_time_cuda();


	HANDLE_ERROR(cudaMalloc((void**)&devBitmap, bitmap.image_size()));

	


	// TODO enable constant memory
	// start_time_cuda();
	// cudaMemcpyToSymbol(spheres, hostSpheres, sizeof(Sphere) * SPHERES);
	// kernel2<<<grids,threads>>>(devBitmap);
	// stop_time_cuda();
	

	// --------------------------------------------
	
	const int SIZE = SPHERES * sizeof(float);
	
	void *r_dev, *g_dev, *b_dev;
	
	float *r_host = (float*) malloc(SIZE);
	float *g_host = (float*) malloc(SIZE);
	float *b_host = (float*) malloc(SIZE);

	for (int i = 0; i < SPHERES; i++) {
		r_host[i] = hostSpheres[i].red;
		g_host[i] = hostSpheres[i].green;
		b_host[i] = hostSpheres[i].blue;
	}
	
	
	HANDLE_ERROR(cudaMalloc((void**)&r_dev, SIZE));
	HANDLE_ERROR(cudaMalloc((void**)&g_dev, SIZE));
	HANDLE_ERROR(cudaMalloc((void**)&b_dev, SIZE));


	HANDLE_ERROR(cudaMemcpy(r_dev, r_host, SIZE, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g_dev, g_host, SIZE, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_dev, b_host, SIZE, cudaMemcpyHostToDevice));

	
	cudaChannelFormatDesc rdesc = cudaCreateChannelDesc<float>();
	cudaBindTexture(NULL, r, r_dev, rdesc, SIZE);
	
	cudaChannelFormatDesc gdesc = cudaCreateChannelDesc<float>();
    cudaBindTexture(NULL, g, g_dev, gdesc, SIZE);

	cudaChannelFormatDesc bdesc = cudaCreateChannelDesc<float>();
	cudaBindTexture(NULL, b, b_dev, bdesc, SIZE);
	

	start_time_cuda();

	kernel3<<<grids,threads>>>(devSpheres, devBitmap);

	stop_time_cuda();

	
	// --------------------------------------------

	free(hostSpheres);
	
	// HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), devBitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
	
	//bitmap.display_and_exit(); //Uncomment if you have a working X session
	
	cudaFree(devBitmap);
	cudaFree(devSpheres);
}
		
	
