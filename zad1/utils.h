#include <fstream>
#include <cstdint>
#include <utility>

using ret_t = std::tuple<uint32_t, uint32_t*, uint32_t*, float*, float*, float>; // V_MAX_IDX, V, E, W, k, m    (W_NUM == E_NUM)

extern float MIN_GAIN;
extern char *FILE_PATH;
extern bool VERBOSE;

__host__
void HandleError(cudaError_t error, const char *file, int line);

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

#define PRINT(begin, end)   thrust::copy((begin), (end), std::ostream_iterator<decltype(*begin)>(std::cout, " "));

#define FULL_MASK 0xffffffff

#define RAW(vec) (thrust::raw_pointer_cast(&(vec)[0]))

#define WARP_SIZE 32
#define SHARED_MEM_SIZE (48 * 1024)



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
void updateSpecific(
    uint32_t* indices, uint32_t indicesNum, 
    const uint32_t* __restrict__  from, 
    uint32_t* __restrict__  to, 
    uint32_t* __restrict__ V);

__host__ 
std::pair<uint16_t, uint16_t> getBlockThreadSplit(uint32_t threads);

__host__
void print_comm_assignment(const uint32_t V_MAX_IDX, const uint32_t* __restrict__ comm);
