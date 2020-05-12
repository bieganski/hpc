#include "utils.h"

__host__ 
float reassign_communities(
                        const uint32_t V_MAX_IDX,
                        uint32_t* __restrict__ V, 
                        uint32_t* __restrict__ E,
                        float*    __restrict__ W,
                        float*    __restrict__ k,
                        float*    __restrict__ ac,
                        uint32_t* __restrict__ comm,
                        uint32_t* __restrict__ newComm,
                        const float m,
                        const float minGain,
                        thrust::device_vector<uint32_t>& globCommAssignment);