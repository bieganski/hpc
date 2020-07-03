#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>
#include <math.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>

#include "hasharray.h"
#include "utils.h"
#include "modopt.h"

using namespace std; // TODO wywalić

using HA = HashArray;

float MIN_GAIN;
char *FILE_PATH;
bool VERBOSE = 0;

uint32_t* V; // vertices
uint32_t* E; // edges
float* W; // weights
float* k; // sum of weights per node
float* ac; // sum of weights per community
uint32_t* comm; // communities
uint32_t* newComm; // new communities after reassign phase
float m; // total sum of weights

uint32_t V_MAX_IDX;



__host__
void debug_print(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V,
                          const uint32_t* __restrict__ E,
                          const float* __restrict__ W,
                          thrust::device_vector<uint32_t>& globalCommAssignment) {        
    printf("\n");                   
    for (int i = 1; i <= V_MAX_IDX; i++) {
        int num = V[i + 1] - V[i];
        if (num == 0)
            continue;
        printf("|%d| > ", i);
        for (int j = 0; j < num; j++) {
            printf("(%d, %f) ", E[V[i] + j], W[V[i] + j]);
        }
        printf("\n");
    }
}

__global__
void compute_size_degree(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V,
                          const uint32_t* __restrict__ comm,
                          uint32_t* __restrict__ commSize,
                          uint32_t* __restrict__ commDegree) {
    int tid = 1 + getGlobalIdx();
    if (tid > V_MAX_IDX)
        return;

    assert(comm[tid] != 0);
    
    atomicAdd(&commSize[comm[tid]], 1);
    atomicAdd(&commDegree[comm[tid]], V[tid + 1] - V[tid]);

}


#define NODE_EXISTS(i, V, E) (V[i+1] - V[i] > 0)

__global__
void compute_compressed_comm(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V,
                          const uint32_t* __restrict__ E, // TODO niepotrzebne
                          const uint32_t* __restrict__ comm,
                          const uint32_t* __restrict__ commSize, // TODO niepotrzebne
                          const uint32_t* __restrict__ vertexStart,
                          uint32_t* __restrict__ tmpCounter,
                          uint32_t* __restrict__ com) {
    int tid = 1 + getGlobalIdx();

    if (tid > V_MAX_IDX)
        return;

    if (!NODE_EXISTS(tid, V, E))
        return;

    int my_comm = comm[tid];

    int idx = atomicAdd(&tmpCounter[my_comm], 1);

    com[vertexStart[my_comm] + idx] = tid;
}

__device__ 
uint32_t CONTRACT_BINS[] = {
    0,
    16,
    384,
    1024,
    2048,
    UINT32_MAX
};

__host__
void computeWTF(const uint32_t* __restrict__ V,
                          thrust::device_vector<uint32_t>& compressedComm,
                          thrust::device_vector<uint32_t>& WTF) {
    thrust::transform(compressedComm.begin(), compressedComm.end(), WTF.begin(), 
        [=] __device__ (const uint32_t& i) {
            return V[i + 1] - V[i];
        });
    thrust::exclusive_scan(WTF.begin(), WTF.end(), WTF.begin());
}


__global__
void compute_comm_neighbors(
    const uint32_t* __restrict__ V,
    const uint32_t* __restrict__ E,
    const float*    __restrict__ W,
    const uint32_t* __restrict__ comm,
    const uint32_t* __restrict__ binCommunities,
    const uint32_t* __restrict__ vertexStart,
    const uint32_t* __restrict__ compressedComm,
    const uint32_t* __restrict__ edgePos,
    const uint32_t* __restrict__ WTF,
    const uint32_t hasharrayEntries,
    uint32_t* __restrict__ newV,
    uint32_t* __restrict__ newE,
    float*    __restrict__ newW,
    uint32_t* __restrict__ freeIndices,
    KeyValueFloat* __restrict__ globalHashtables
) {
    
    extern __shared__ char hashtables[];

    int myCommPtr = blockIdx.x;
    int myEdgePtr = threadIdx.x;

    uint32_t myComm = binCommunities[myCommPtr];

    uint32_t firstNodePtrIncl = vertexStart[myComm];
    uint32_t lastNodePtrExcl  = vertexStart[myComm + 1];

    char* hasharr_ptr;

    if (globalHashtables == nullptr) {
        hasharr_ptr = hashtables;
    } else {
        hasharr_ptr = (char*) (&globalHashtables[myCommPtr * (2 * hasharrayEntries)]);
    }

    KeyValueFloat* hashWeight = (KeyValueFloat*) hasharr_ptr;
    KeyValueInt*   hashComm   = (KeyValueInt*)   hashWeight + hasharrayEntries;

    for (int i = myEdgePtr; i < hasharrayEntries; i += WARP_SIZE) {
        hashWeight[i] = {.key = hashArrayNull, .value = (float) 0}; // 0 for easy atomicAdd
        hashComm[i]   = {.key = hashArrayNull, .value = hashArrayNull};
    }

    uint32_t insertedByMe = 0;
    uint32_t start = firstNodePtrIncl;
    uint32_t offset = WTF[firstNodePtrIncl];

    bool finish = false; // cannot use early return because of usage of warp-level primitives

    uint32_t nodeIdx = start + 0;
    while(true) {

        if (nodeIdx >= lastNodePtrExcl) {
            break;
        }

        uint32_t myNode = compressedComm[nodeIdx]; // common for whole warp
        // myNode :: oldVNum 
        uint32_t myNeighNum = V[myNode + 1] - V[myNode]; // :: size(oldVnum)
        assert(myNeighNum == WTF[nodeIdx + 1] - WTF[nodeIdx]);
        
        for (int i = 0; i < ceil(((float) myNeighNum) / 32.0); i++) {
            uint32_t myEdgeIdx = threadIdx.x + i * 32; // :: Epos

            if (myEdgeIdx >= myNeighNum) {
                break;
            }

            // V :: Vnum -> Epos
            // E :: Epos -> Vnum
            // W :: Epos -> W(Vnum)
            // comm :: Vnum -> Cnum
            // CC :: NVnum -> VNum
            // edgePos :: NCnum -> NEpos

            // WTF :: NVnum -> size(VNum)

            uint32_t myEdgeVnum = E[V[myNode] + myEdgeIdx];
            uint32_t myEdgeWeight = W[V[myNode] + myEdgeIdx];

            if ( HA::insertWithFeedback(hashComm, hashWeight, comm[myEdgeVnum], comm[myEdgeVnum], 
                    myEdgeWeight, hasharrayEntries) ) {
                insertedByMe++;
            } else {
                ;
            }
        }

        nodeIdx++;
    }

    __syncthreads();


    /*
    while(true) {

        // looking for my node and edge
        uint32_t myEdge = -1;
        uint32_t myNode = -1;
        uint32_t edgeIdx = -1;

        for (uint32_t i = start; !finish && (i < lastNodePtrExcl); i++) {
            if (myEdgePtr < WTF[i + 1] - offset) {
                myNode = compressedComm[i];
                start = i; // for next iteration
                edgeIdx = myEdgePtr - (WTF[i] - offset);
                myEdge = E[V[myNode] + edgeIdx];
                break;
            } else if (i == lastNodePtrExcl - 1) {
                // they don't need me
                finish = true;
            }
        }

        if (finish)
            break;

        // I know who am I, now add my neighbor to sum of weights
        if ( HA::insertWithFeedback(hashComm, hashWeight, comm[myEdge], comm[myEdge], W[V[myNode] + edgeIdx], hasharrayEntries) ) {
            insertedByMe++;
        } else {
            ;
        }

        myEdgePtr += WARP_SIZE;
    } // while(true)
    */


    // now, compute number of totally inserted in this warp's community
    
    int mask = __ballot_sync(0xFFFFFFFF, 1);

    assert(mask == FULL_MASK);

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        insertedByMe += __shfl_down_sync(mask, insertedByMe, offset); // only warp with idx == 0 keeps proper value
    
    // assert(0 == __ffs(mask) - 1);
    assert(__ffs(mask) & 1);
    // int leader = 0; // __ffs(mask) - 1; // = 0 // TODO assumption: zero-idx-thread is alive

    uint32_t commNeighborsNum = __shfl_sync(mask, insertedByMe, 0);

    int myEdgePtr0 = threadIdx.x;
    if (myEdgePtr0 == 0) {
        // WARNING: we use old community id, because we already know free E indices!
        newV[myComm] = commNeighborsNum; // will be computed prefix sum on it later
    }

    // assert(mask == FULL_MASK);
    uint32_t idx0 = edgePos[myComm];
    
    if (threadIdx.x != 0)
        return;
    // TODO
    // this should be performed by all warps, but there were unknown
    // problems with this
    // for (int i = myEdgePtr0; i < hasharrayEntries; i += WARP_SIZE) {
    for (int i = 0; i < hasharrayEntries; i++) {
        if (hashComm[i].key != hashArrayNull) {
            uint32_t myIdx = atomicAdd(&freeIndices[myComm], 1);
            newE[idx0 + myIdx] = hashComm[i].value;
            
            if (myComm == hashComm[i].value) {
                // TODO self-loop, should it be halved?
                newW[idx0 + myIdx] = hashWeight[i].value; //  / 2.0;
            } else {
                newW[idx0 + myIdx] = hashWeight[i].value;
            }
        }
    }
    assert(freeIndices[myComm] == newV[myComm]);
}


__global__
void compute_k(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V,
                          const uint32_t* __restrict__ E,
                          const float*    __restrict__ W,
                          float*          __restrict__ k) {
    
    int tid = 1 + getGlobalIdx();
    if (tid > V_MAX_IDX)
        return;
    int idx0 = V[tid];
    int num = V[tid + 1] - idx0;
    k[tid] = 0;
    for (int i = 0; i < num; i++) {
        k[tid] += E[idx0 + i] == tid ? 
            W[idx0 + i] //  / 2.0 TODO ?
            : W[idx0 + i];
    }
}

__global__
void recompute_globalCommAssignment(
        const uint32_t V_MAX_IDX,
        const uint32_t* __restrict__ newID,
        const uint32_t* __restrict__ comm,
        uint32_t* __restrict__ globalCommAssignment) {

    int tid = 1 + getGlobalIdx();
    if (tid > V_MAX_IDX)
        return;

    if (comm[tid] != tid) {
        // I was moved to another community
        globalCommAssignment[tid] = comm[tid];
    }
    cudaDeviceSynchronize(); // avoid races

    if (globalCommAssignment[globalCommAssignment[tid]] != 0) {
        globalCommAssignment[tid] = globalCommAssignment[globalCommAssignment[tid]];
    }
}


bool DBG = false;

template <typename T>
void print_DEBUG(uint32_t max_idx, T* arr, const char* name, bool verbose = false, bool from_zero = false) {
    if (!DBG)
        return;
    T* mem = (T*) malloc((max_idx + 1) * sizeof(T));
    cudaMemcpy((void*) mem, (void*) arr, (max_idx + 1) * sizeof(T), cudaMemcpyDeviceToHost);
    if (!verbose) {
        printf("[C]: %s[1]: %d\n", name, mem[1]);
        printf("[C]: %s[5]: %d\n", name, mem[max_idx]);
    } else {
        int i  = from_zero == false ? 1 : 0;
        printf("[C]: %s[%d,%d]: ", name, i, max_idx);
        for (; i <= max_idx; i++) {
            printf(" %d", mem[i]);
        }
        printf("\n");
    }
    free((void*) mem);
}


__host__
void contract(const uint32_t V_MAX_IDX,
                          uint32_t* __restrict__ V, 
                          uint32_t* __restrict__ E,
                          float*    __restrict__ W,
                          float*    __restrict__ k,
                          const uint32_t* __restrict__ comm,
                          thrust::device_vector<uint32_t>& globalCommAssignment) {

    // TODO przenieśc je wyżej, żeby alokować tylko raz
    thrust::device_vector<uint32_t> commSize(V_MAX_IDX + 1, (uint32_t) 0);
    thrust::device_vector<uint32_t> commDegree(V_MAX_IDX + 1, (uint32_t) 0);
    thrust::device_vector<uint32_t> edgePos(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> newID(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> vertexStart(V_MAX_IDX + 2, 0);
    thrust::device_vector<uint32_t> tmpCounter(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> compressedComm(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> commSeq(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> freeIndices(V_MAX_IDX + 1, 0);

    thrust::device_vector<uint32_t> WTF(V_MAX_IDX + 1 , 0);

    thrust::sequence(commSeq.begin(), commSeq.end());

    auto pair = getBlockThreadSplit(V_MAX_IDX);

    compute_size_degree<<<pair.first, pair.second>>> (V_MAX_IDX, V, comm, RAW(commSize), RAW(commDegree));

    cudaDeviceSynchronize();

    thrust::transform(commSize.begin(), commSize.end(), newID.begin(), [] __device__ (const uint32_t& size) {return size > 0 ? 1 : 0;});
    thrust::inclusive_scan(newID.begin(), newID.end(), newID.begin());
    
    thrust::inclusive_scan(commSize.begin(), commSize.end(), &vertexStart[1]); // start output at 1 
    
    compute_compressed_comm <<<pair.first, pair.second>>> (V_MAX_IDX, V, E, comm, 
            RAW(commSize), RAW(vertexStart), RAW(tmpCounter), RAW(compressedComm));

    // compressedCom <=> `com` from paper
    cudaDeviceSynchronize();

    print_DEBUG(V_MAX_IDX + 1, V, "V", true);
    print_DEBUG(V[V_MAX_IDX + 1] - 1, E, "E", true);
    print_DEBUG(V_MAX_IDX, comm, "comm", true);
    

    print_DEBUG(V_MAX_IDX, RAW(commSize), "commSize", true);
    print_DEBUG(V_MAX_IDX, RAW(commDegree), "commDegree", true);

    thrust::exclusive_scan(commDegree.begin(), commDegree.end(), edgePos.begin());

    auto commDegreeLambda = RAW(commDegree); // you cannot use thrust's vector in device code

    auto partitionGenerator = [=](int rightIdx) {
        return [=] __device__ (const uint32_t& i) {
            return commDegreeLambda[i] <= CONTRACT_BINS[rightIdx];
        };
    };

    uint32_t* contractBinsHost = (uint32_t*) malloc(sizeof(CONTRACT_BINS));
    cudaMemcpyFromSymbol(contractBinsHost, CONTRACT_BINS, sizeof(CONTRACT_BINS), 0, cudaMemcpyDeviceToHost);

    computeWTF(V, compressedComm, WTF);

    print_DEBUG(V_MAX_IDX, RAW(compressedComm), "compressedComm", true, true);
    print_DEBUG(V_MAX_IDX, RAW(WTF), "WTF", true);
    print_DEBUG(V_MAX_IDX, RAW(edgePos), "edgePos", true);
    print_DEBUG(V_MAX_IDX, RAW(vertexStart), "vertexStart", true);

    uint32_t E_size = V[V_MAX_IDX + 1];
    // Each call to `compute_comm _neighbors` kernel updates part of these values
    thrust::device_vector<uint32_t> newV(V_MAX_IDX + 2, 0);
    thrust::device_vector<uint32_t> newE(E_size, 0);
    thrust::device_vector<float>    newW(E_size, 0);

    // we don't want empty communities
    auto it0 = thrust::partition(commSeq.begin(), commSeq.end(), partitionGenerator(0));
    
    for (int i = 1; ; i++) {

        auto it = thrust::partition(it0, commSeq.end(), partitionGenerator(i));

        uint32_t binNodesNum = thrust::distance(it0, it);
        if (binNodesNum == 0)
            break;
            
        // handle only communities with same degree boundary
        uint32_t degUpperBound = contractBinsHost[i];

        KeyValueFloat* globalHashArrayPtr;
        uint32_t shmSize;
        uint32_t hashArrayEntriesPerComm;

        if (degUpperBound == UINT32_MAX) {
            hashArrayEntriesPerComm = next_2_pow(20000); // TODO customize this

            shmSize = 0;
            uint32_t TOTAL_HASHARRAY_ENTRIES = (2 * hashArrayEntriesPerComm) * binNodesNum;

            HANDLE_ERROR(cudaHostAlloc((void**)&globalHashArrayPtr, sizeof(KeyValueFloat) * TOTAL_HASHARRAY_ENTRIES, cudaHostAllocDefault));
            std::memset(globalHashArrayPtr, '\0', sizeof(KeyValueFloat) * TOTAL_HASHARRAY_ENTRIES);
            HANDLE_ERROR(cudaHostGetDevicePointer(&globalHashArrayPtr, globalHashArrayPtr, 0));

        } else {
            hashArrayEntriesPerComm = next_2_pow(degUpperBound); // TODO customize this

            shmSize = sizeof(KeyValueInt) * (2 * hashArrayEntriesPerComm); // one block per community

            assert(shmSize <= SHARED_MEM_SIZE);

            globalHashArrayPtr = nullptr;
        }

        compute_comm_neighbors <<< binNodesNum, WARP_SIZE, shmSize >>> 
            (
                V, E, W, comm, 
                RAW(it0),
                RAW(vertexStart), 
                RAW(compressedComm), 
                RAW(edgePos), 
                RAW(WTF), 
                hashArrayEntriesPerComm, 
                RAW(newV), 
                RAW(newE), 
                RAW(newW), 
                RAW(freeIndices),
                globalHashArrayPtr
            );


        cudaDeviceSynchronize();        

        it0 = it; // it0 points to first node that wasn't processed yet
    }

    thrust::exclusive_scan(newV.begin(), newV.end(), newV.begin());

    thrust::device_vector<uint32_t> realNewE(newE.size(), 0);
    thrust::device_vector<float> realNewW(newW.size(), 0);

    thrust::copy_if(newE.begin(), newE.end(), realNewE.begin(), [] __device__ (const uint32_t& x) {return x != 0;});
    thrust::copy_if(newW.begin(), newW.end(), realNewW.begin(), [] __device__ (const float& x) {return x != 0;});


    if (VERBOSE) {
        auto ppair = getBlockThreadSplit(V_MAX_IDX);
        recompute_globalCommAssignment<<<ppair.first, ppair.second>>>(V_MAX_IDX, RAW(newID), comm, RAW(globalCommAssignment));

        cudaDeviceSynchronize();
    }

    HANDLE_ERROR(cudaMemcpy((void*)V, (void*)RAW(newV), newV.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy((void*)E, (void*)RAW(realNewE), realNewE.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy((void*)W, (void*)RAW(realNewW), realNewW.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    thrust::host_vector<uint32_t> seq(V_MAX_IDX + 1);
    thrust::sequence(seq.begin(), seq.end());
    HANDLE_ERROR(cudaMemcpy((void*)comm, (void*)RAW(seq), (V_MAX_IDX + 1) * sizeof(uint32_t), cudaMemcpyHostToHost));
    
    cudaDeviceSynchronize();
    auto k_pair = getBlockThreadSplit(V_MAX_IDX);
    compute_k<<<k_pair.first, k_pair.second>>> (V_MAX_IDX, V, E, W, k);

    cudaDeviceSynchronize();

    free(contractBinsHost);
}



/**
 * UWAGA:
 * N - rozmiar tablicy k
 * N + 1 - rozmiar tablicy V
 * V - iterujemy od 1
 * E, W - rozmiar 2*N, iterujemy od 0
 * */
int main(int argc, char **argv) {
    if (parse_args(argc, argv)) {
        exit(1);
    }
    
    ret_t res = parse_inut_graph(get_input_content());

    V_MAX_IDX = std::get<0>(res);
    m = std::get<5>(res);

    cudaEvent_t startCopy, stopCopy, startComp, stopComp;
    cudaEventCreate(&startCopy);
    cudaEventCreate(&stopCopy);
    cudaEventCreate(&startComp);
    cudaEventCreate(&stopComp);

    cudaEventRecord( startCopy, 0 );


    HANDLE_ERROR(cudaHostGetDevicePointer(&V, std::get<1>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&E, std::get<2>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&W, std::get<3>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&k, std::get<4>(res), 0));

    uint32_t* tmp;
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(uint32_t) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&newComm, tmp, 0));
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(float) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&ac, tmp, 0));

    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(uint32_t) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    for (int i = 0; i <= V_MAX_IDX; i++) {
        tmp[i] = i; // each node is in it's own community at the beginning
    }
    HANDLE_ERROR(cudaHostGetDevicePointer(&comm, tmp, 0));

    cudaEventRecord(stopCopy, 0);
    cudaEventSynchronize(stopCopy);
    float elapsedTimeCopy;
    cudaEventElapsedTime( &elapsedTimeCopy, startCopy, stopCopy);
    cudaEventDestroy(startCopy);
    cudaEventDestroy(stopCopy);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte); // TODO customize
    // cudaFuncSetCacheConfig(reassign_nodes, cudaFuncCachePreferShared); // TODO

    cudaDeviceSynchronize();

    thrust::device_vector<uint32_t> globCommAssignment(V_MAX_IDX + 1);

    cudaEventRecord( startComp, 0 );
    float mod = reassign_communities(V_MAX_IDX, V, E, W, k, ac, comm, newComm, m, MIN_GAIN, globCommAssignment);
    cudaEventRecord( stopComp, 0 );

    cudaEventSynchronize(stopComp);
    float elapsedTimeComp;
    cudaEventElapsedTime( &elapsedTimeComp, startComp, stopComp);
    cudaEventDestroy(startComp);
    cudaEventDestroy(stopComp);

    printf("%f\n", mod);
    printf("%f %f\n", elapsedTimeComp, elapsedTimeCopy);

    thrust::host_vector<uint32_t> resComm(V_MAX_IDX + 1);
    
    thrust::copy(globCommAssignment.begin(), globCommAssignment.end(), resComm.begin());


    if (VERBOSE) {
        print_comm_assignment(V_MAX_IDX, RAW(resComm));
    }
    // TODO cleanup
    // cudaFree(V);
    return 0;
}
