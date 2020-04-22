#include <fstream>
#include <cstdint>

using ret_t = std::tuple<uint32_t, uint32_t*, uint32_t*, float*, float*>; // V_MAX_IDX, V, E, W, k    (W_NUM == E_NUM)

extern float MIN_GAIN;
extern char *FILE_PATH;
extern bool VERBOSE;

static void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

int parse_args (int argc, char **argv);

std::ifstream get_input_content();

ret_t parse_inut_graph(std::ifstream content);

__device__ uint32_t next_2_pow(uint32_t n);