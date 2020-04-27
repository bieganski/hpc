#include <fstream>
#include <cstdint>
#include <utility>

using ret_t = std::tuple<uint32_t, uint32_t*, uint32_t*, float*, float*, float>; // V_MAX_IDX, V, E, W, k, m    (W_NUM == E_NUM)

extern float MIN_GAIN;
extern char *FILE_PATH;
extern bool VERBOSE;

__host__
static void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

#define PRINT(begin, end)   thrust::copy((begin), (end), std::ostream_iterator<decltype(*begin)>(std::cout, " "));

#define FULL_MASK 0xffffffff


__host__
int parse_args (int argc, char **argv);

__host__
std::ifstream get_input_content();

__host__
ret_t parse_inut_graph(std::ifstream content);

__host__ __device__ 
uint32_t next_2_pow(uint32_t n);


__device__ 
void binprintf(uint32_t v);

__device__
int getGlobalIdx();

__global__ 
void updateSpecific(uint32_t* indices, uint32_t indicesNum, uint32_t* from, uint32_t* to);

__host__ 
std::pair<uint16_t, uint16_t> getBlockThreadSplit(uint32_t threads);