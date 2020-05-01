#include "utils.h"


// __global__
// void computeAC(const float* __restrict__ k,
//                float*       __restrict__ ac,
//                uint32_t*    __restrict__ comm);

// __host__
// void zeroAC(float* ac);

// /**
//  * This kernel function converts multiple nodes,
//  * each node got it's own part of shared memory for node-common data, i.a hashArrays.
//  */
// __global__ 
// void reassign_nodes(
//                         const uint32_t  numNodes,
//                         const uint32_t* __restrict__ binNodes,
//                         const uint32_t* __restrict__ V,
//                         const uint32_t* __restrict__ E,
//                         const float*    __restrict__ W,
//                         const float*    __restrict__ k,
//                         const float*    __restrict__ ac,
//                         const uint32_t* __restrict__ comm,
//                         uint32_t*       __restrict__ newComm,
//                         const uint32_t maxDegree,
//                         const uint32_t nodesPerBlock,
//                         const uint32_t hasharrayEntries,
//                         const float m,
//                         const float minGain);

// __host__ 
// float computeMod(float ei_to_Ci_sum, float m, const float* ac);

// __host__ float reassign_communities_bin(
//                         const uint32_t* binNodes,
//                         const uint32_t binNodesNum,
//                         const uint32_t* __restrict__ V, 
//                         const uint32_t* __restrict__ E,
//                         const float*    __restrict__ W,
//                         const float*    __restrict__ k,
//                         const float*    __restrict__ ac,
//                         uint32_t* __restrict__ comm,
//                         uint32_t* __restrict__ newComm,
//                         const uint32_t maxDegree,
//                         const float m,
//                         const float minGain);

// __global__ void computeEiToCi(float*    __restrict__ ei_to_Ci,
//                          const uint32_t* __restrict__ V,
//                          const uint32_t* __restrict__ E,
//                          const float*    __restrict__ W,
//                          const uint32_t* __restrict__ comm);


__host__ 
float reassign_communities(
                        const uint32_t V_MAX_IDX,
                        const uint32_t* __restrict__ V, 
                        const uint32_t* __restrict__ E,
                        const float*    __restrict__ W,
                        const float*    __restrict__ k,
                        float*    __restrict__ ac,
                        uint32_t* __restrict__ comm,
                        uint32_t* __restrict__ newComm,
                        float*    __restrict__ ei_to_Ci,
                        const float m,
                        const float minGain);