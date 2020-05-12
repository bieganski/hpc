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
    // printf("******************  GLOBALE: \n");
    // thrust::device_vector<uint32_t> tmp(globalCommAssignment.size());
    // thrust::sequence(tmp.begin(), tmp.end());
    // thrust::copy(tmp.begin(), tmp.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
    // std::cout << "\n";
    // thrust::copy(globalCommAssignment.begin(), globalCommAssignment.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
    // printf("****************************************************\n");
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

    // __syncthreads(); // TODO - wywalić

    // if (tid == 1) {
    //     printf("WYPISUJĘ OBLICZONE COMMUNITY SIZES: \n");
    //     for (int i = 0; i <=V_MAX_IDX; i++) {
    //         printf("%d ", commSize[i]); 
    //     }
    //     printf("\n");


    //     printf("WYPISUJĘ OBLICZONE COMMUNITY DEGREES: \n");
    //     for (int i = 0; i <=V_MAX_IDX; i++) {
    //         printf("%d ", commDegree[i]); 
    //     }
    //     printf("\n");
    // }
}


#define NODE_EXISTS(i, V, E) (V[i+1] - V[i] > 0)

__global__
void compute_compressed_comm(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V,
                          const uint32_t* __restrict__ E,
                          const uint32_t* __restrict__ comm,
                          uint32_t* __restrict__ commSize,
                          uint32_t* __restrict__ vertexStart,
                          uint32_t* __restrict__ tmpCounter,
                          uint32_t* __restrict__ compressedComm) {
    int tid = 1 + getGlobalIdx();

    if (tid > V_MAX_IDX)
        return;

    if (!NODE_EXISTS(tid, V, E))
        return;

    int my_comm = comm[tid];

    int idx = atomicAdd(&tmpCounter[my_comm], 1);

    compressedComm[vertexStart[my_comm] + idx] = tid;
}

__device__ uint32_t CONTRACT_BINS[] = {
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

    int myCommPtr = blockIdx.x; // threadIdx.y + (blockIdx.y * blockDim.y);
    int myEdgePtr = threadIdx.x; //  + (blockIdx.x * blockDim.x);

    uint32_t myComm = binCommunities[myCommPtr];

    // printf("MYCOM: %d\n", myComm);

    uint32_t firstNodePtrIncl = vertexStart[myComm];
    uint32_t lastNodePtrExcl  = vertexStart[myComm + 1];

    char* hasharr_ptr;

    if (globalHashtables == nullptr) {
        hasharr_ptr = hashtables;
        // printf("WYBOR: uzywam shared mem!\n");
    } else {
        int tmp = (int) (hasharr_ptr - (char*)globalHashtables);
        hasharr_ptr = (char*) (&globalHashtables[myCommPtr * (2 * hasharrayEntries)]);
        // printf("GAS2: p=%p, n=%x\n", (void*) hasharr_ptr, myCommPtr * (2 * hasharrayEntries));
        // printf("WYBOR: uzywam global mem pod offsetem %d = %d * (2 * %d), diff=d, %p, %p, lol: d\n", myCommPtr * (2 * hasharrayEntries), 
        //                                                                              myCommPtr ,hasharrayEntries, 
        //                                                                             // tmp,
        //                                                                             //  hasharr_ptr - (char*)globalHashtables,
        //                                                                              (void*) hasharr_ptr, (void*)globalHashtables
                                                                                    
        //                                                                             //  __popcll((unsigned long long)((void*) hasharr_ptr - (void*) globalHashtables))
        //                                                                              );
    }

    KeyValueFloat* hashWeight = (KeyValueFloat*) hasharr_ptr;
    KeyValueInt*   hashComm   = (KeyValueInt*)   hashWeight + hasharrayEntries;

    for (int i = myEdgePtr; i < hasharrayEntries; i += WARP_SIZE) {
        hashWeight[i] = {.key = hashArrayNull, .value = (float) 0}; // 0 for easy atomicAdd
        hashComm[i]   = {.key = hashArrayNull, .value = hashArrayNull};
    }

    uint32_t insertedByMe = 0;
    uint32_t start = firstNodePtrIncl;
    uint32_t offset = WTF[firstNodePtrIncl]; // TODO benchmark bez tego

    bool finish = false; // cannot use early return because of usage of warp-level primitives
    while(true) {

        // looking for my node and edge
        uint32_t myEdge = -1;
        uint32_t myNode = -1;
        uint32_t edgeIdx = -1;

        // printf("debug: %d, %d\n", start, lastNodePtrExcl);
        // if (start >= lastNodePtrExcl)
        //     return ;
        for (uint32_t i = start; !finish && (i < lastNodePtrExcl); i++) {
            if (myEdgePtr < WTF[i + 1] - offset) {
                myNode = compressedComm[i];
                start = i; // for next iteration
                edgeIdx = myEdgePtr - (WTF[i] - offset);
                myEdge = E[V[myNode] + edgeIdx];
                // printf("%d: dla myPtrEdge: %d znalazlem edge %d (edgeIdx = %d)\n", myNode, myEdgePtr, myEdge, edgeIdx);
                break;
            } else if (i == lastNodePtrExcl - 1) {
                // they don't need me :(
                // printf("wychodze bo mnie nie potrzebują: %d, %d\n", myComm, myEdgePtr);
                finish = true;
            }
        }

        if (finish)
            break;

        // I know who am I, now add my neighbor to sum of weights
        // printf( "%d->%d: dodaje do haszarray wage %f, entries: %d\n", myNode, myEdge, W[V[myNode] + edgeIdx], hasharrayEntries);
        if ( HA::insertWithFeedback(hashComm, hashWeight, comm[myEdge], comm[myEdge], W[V[myNode] + edgeIdx], hasharrayEntries) ) {
            insertedByMe++;
        } else {
            ;
        }

        myEdgePtr += WARP_SIZE;
        // printf("%d: inserted by me: %d\n", myEdgePtr % WARP_SIZE, insertedByMe);
    } // while(true)

    // now, compute number of totally inserted in this warp's community
    
    int mask = __ballot_sync(0xFFFFFFFF, 1);

    if (myEdgePtr == 0) {
        printf("MASK: ");
        binprintf(mask);
        printf("\n");
    }

    assert(mask == FULL_MASK);

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        insertedByMe += __shfl_down_sync(mask, insertedByMe, offset); // only warp with idx == 0 keeps proper value

    
    // assert(0 == __ffs(mask) - 1);

    assert(__ffs(mask) & 1);
    int leader = 0; // __ffs(mask) - 1; // = 0 // TODO assumption: zero-idx-thread is alive

    uint32_t commNeighborsNum = __shfl_sync(mask, insertedByMe, 0);

    int myEdgePtr0 = threadIdx.x;
    if (myEdgePtr0 == 0) {
        // WARNING: we use old community id, because we already know free E indices!
        newV[myComm] = commNeighborsNum; // will be computed prefix sum on it later
    }

    // assert(mask == FULL_MASK); // !!!!!
    uint32_t idx0 = edgePos[myComm];
    // if (myEdgePtr0 == 0) {
    //     printf("%d: our idx0: %d\n", myComm, idx0);
    // }
    if (threadIdx.x != 0)
        return;
    printf("LOL: (%d) -> %d, uniq edgePtr0: %d\n", myCommPtr, myComm, myEdgePtr0);
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
            
            // printf("%d: dodaje sasiada %d pod idx %d\n", myComm, hashComm[i].value, idx0 + myIdx); 
        }
    }
    printf("LOL2:(%d) ->  %d vs %d\n", myCommPtr, freeIndices[myComm] ,newV[myComm]);
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

    // assert(blockIdx.x * blockIdx. y * blockIdx.z == 0); // only one block supported, because of __syncthreads() call
    // assert(newID[0] == 0); // TODO comment out

    int tid = 1 + getGlobalIdx();
    if (tid > V_MAX_IDX)
        return;

    if (comm[tid] != tid) {
        // I was moved to another community
        globalCommAssignment[tid] = comm[tid];
    }
    cudaDeviceSynchronize();

    if (globalCommAssignment[globalCommAssignment[tid]] != 0) {
        globalCommAssignment[tid] = globalCommAssignment[globalCommAssignment[tid]];
    }



    // uint32_t tmp = globalCommAssignment[tid];
    // __syncthreads();
    // if (newID[tid] > newID[tid - 1]) {
    //     globalCommAssignment[newID[tid]] = tmp;
    // }
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
    thrust::device_vector<uint32_t> commSize(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> commDegree(V_MAX_IDX + 1, 0);
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
    // printf("newID: \n");
    // // PRINT(newID.begin(), newID.end());
    // thrust::copy(newID.begin(),newID.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
    // printf("\n");

    compute_compressed_comm <<<pair.first, pair.second>>> (V_MAX_IDX, V, E, comm, 
            RAW(commSize), RAW(vertexStart), RAW(tmpCounter), RAW(compressedComm));


    cudaDeviceSynchronize();



    // printf("VERTEX START: \n");
    // thrust::copy(vertexStart.begin(), vertexStart.end(), std::ostream_iterator<uint32_t>(std::cout, " "));


    // printf("\nCOMPRESSED COMM: \n");
    // thrust::copy(compressedComm.begin(), compressedComm.end(), std::ostream_iterator<uint32_t>(std::cout, " "));

    thrust::exclusive_scan(commDegree.begin(), commDegree.end(), edgePos.begin());

    // printf("EDGE POS: \n");
    // thrust::copy(edgePos.begin(), edgePos.end(), std::ostream_iterator<uint32_t>(std::cout, " "));


    auto commDegreeLambda = RAW(commDegree); // you cannot use thrust's vector in device code

    auto partitionGenerator = [=](int rightIdx) {
        return [=] __device__ (const uint32_t& i) {
            return commDegreeLambda[i] <= CONTRACT_BINS[rightIdx];
        };
    };

    // TODO to też powinno być na zewnątrz
    // TODO free zrobione
    uint32_t* contractBinsHost = (uint32_t*) malloc(sizeof(CONTRACT_BINS));
    cudaMemcpyFromSymbol(contractBinsHost, CONTRACT_BINS, sizeof(CONTRACT_BINS), 0, cudaMemcpyDeviceToHost);

    computeWTF(V, compressedComm, WTF);

    // printf("WTF: \n");
    // // PRINT(WTF.begin(), WTF.end());
    // thrust::copy(WTF.begin(), WTF.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
    // printf("\n");

    uint32_t E_size = V[V_MAX_IDX + 1];
    // printf("E size: %d\n", E_size);
    // Each call to `compute_comm _neighbors` kernel updates part of these values
    thrust::device_vector<uint32_t> newV(V_MAX_IDX + 2, 0);
    thrust::device_vector<uint32_t> newE(E_size, 0);
    thrust::device_vector<float>    newW(E_size, 0);

    // we don't want empty communities
    auto it0 = thrust::partition(commSeq.begin(), commSeq.end(), partitionGenerator(0));
    
    for (int i = 1; ; i++) {

        auto it = thrust::partition(it0, commSeq.end(), partitionGenerator(i));

        uint32_t binNodesNum = thrust::distance(it0, it);
        printf("BIN NODES NODES : %d\n", binNodesNum);
        if (binNodesNum == 0)
            break;
            
        // handle only communities with same degree boundary
        uint32_t degUpperBound = contractBinsHost[i];

        KeyValueFloat* globalHashArrayPtr;
        uint32_t shmSize;
        uint32_t hashArrayEntriesPerComm;

        if (degUpperBound == UINT32_MAX) {
            hashArrayEntriesPerComm = next_2_pow(100000); // TODO customize this

            shmSize = 0;
            uint32_t TOTAL_HASHARRAY_ENTRIES = (2 * hashArrayEntriesPerComm) * binNodesNum;

            // thrust::device_vector<KeyValueFloat> globalHashArray(TOTAL_HASHARRAY_ENTRIES, 0);
            // KeyValueFloat* globalHashArray; 
            HANDLE_ERROR(cudaHostAlloc((void**)&globalHashArrayPtr, sizeof(KeyValueFloat) * TOTAL_HASHARRAY_ENTRIES, cudaHostAllocDefault));
            std::memset(globalHashArrayPtr, '\0', sizeof(KeyValueFloat) * TOTAL_HASHARRAY_ENTRIES);
            HANDLE_ERROR(cudaHostGetDevicePointer(&globalHashArrayPtr, globalHashArrayPtr, 0));
            // globalHashArrayPtr = globalHashArray; // RAW(globalHashArray);
            // printf("GAS: p_beg=%p n=%x, \n", (void*) globalHashArrayPtr, TOTAL_HASHARRAY_ENTRIES);
            // printf("GAS: p_end=%p \n", (void*) &globalHashArrayPtr[TOTAL_HASHARRAY_ENTRIES]);

        } else {
            hashArrayEntriesPerComm = next_2_pow(degUpperBound); // TODO customize this

            shmSize = sizeof(KeyValueInt) * (2 * hashArrayEntriesPerComm); // one block per community

            assert(shmSize <= SHARED_MEM_SIZE);

            globalHashArrayPtr = nullptr;
        }

        printf("MOD contract: binnodes: %d, global: %d, shm_size: %d, hashentries: %d\n", binNodesNum, globalHashArrayPtr != nullptr, shmSize, hashArrayEntriesPerComm);

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

        printf("********************************************************   END OF BIN with MAXDEG=%d\n", degUpperBound);
        it0 = it; // it0 points to first node that wasn't processed yet
    }

    //   printf("\newID:");
    //     thrust::copy(newID.begin(), newID.end(), 
    //         std::ostream_iterator<uint32_t>(std::cout, " "));
    //     printf("\nnowe V:");
    //     thrust::copy(newV.begin(), newV.end(), 
    //         std::ostream_iterator<uint32_t>(std::cout, " "));
        // printf("\nnowe E:");
        // thrust::copy(newE.begin(), newE.end(), 
        //     std::ostream_iterator<uint32_t>(std::cout, " "));
    //     printf("\nnowe W:");
    //     thrust::copy(newW.begin(), newW.end(), 
    //         std::ostream_iterator<float>(std::cout, " "));
    
    // OK, all the communities have been restructured, now let's merge the results

    thrust::exclusive_scan(newV.begin(), newV.end(), newV.begin());

    thrust::device_vector<uint32_t> realNewE(newE.size(), 0);
    thrust::device_vector<float> realNewW(newW.size(), 0);

    thrust::copy_if(newE.begin(), newE.end(), realNewE.begin(), [] __device__ (const uint32_t& x) {return x != 0;});
    thrust::copy_if(newW.begin(), newW.end(), realNewW.begin(), [] __device__ (const float& x) {return x != 0;});


    if (VERBOSE) {
        // dim3 __dim((uint) ceil((float)V_MAX_IDX / 1024.0) , min(V_MAX_IDX, 1024));
        auto ppair = getBlockThreadSplit(V_MAX_IDX);
        recompute_globalCommAssignment<<<ppair.first, ppair.second>>>(V_MAX_IDX, RAW(newID), comm, RAW(globalCommAssignment));

        cudaDeviceSynchronize();
    }
    
    // printf("COMMUNITY MAPPING: \n");
    // thrust::copy(commSeq.begin(), commSeq.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
    // std::cout << "\n";
    // thrust::copy(globalCommAssignment.begin(), globalCommAssignment.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
    // std::cout << "\n";

    // thrust::transform(newV.begin(), newV.end(), thrust::make_discard_iterator(), [=] __device__(const uint32_t& x) {assert(E[x] != 0); printf("haha\n");return 0;});

    HANDLE_ERROR(cudaMemcpy((void*)V, (void*)RAW(newV), newV.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy((void*)E, (void*)RAW(realNewE), realNewE.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy((void*)W, (void*)RAW(realNewW), realNewW.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // TODO check this
    thrust::host_vector<uint32_t> seq(V_MAX_IDX + 1);
    thrust::sequence(seq.begin(), seq.end());
    HANDLE_ERROR(cudaMemcpy((void*)comm, (void*)RAW(seq), (V_MAX_IDX + 1) * sizeof(uint32_t), cudaMemcpyHostToHost));

    // printf("COMMMMMMMMMMM : \n");
    // for (int i = 1; i <= V_MAX_IDX; i++) {
    //     printf("%d ", comm[i]);
    // } 
    // printf("\n");

    
    cudaDeviceSynchronize();
    auto k_pair = getBlockThreadSplit(V_MAX_IDX);
    compute_k<<<k_pair.first, k_pair.second>>> (V_MAX_IDX, V, E, W, k);

    cudaDeviceSynchronize();

    // printf("KKKKKKK : \n");
    // for (int i = 1; i <= V_MAX_IDX; i++) {
    //     printf("%f ", k[i]);
    // }
    printf("\n");
    printf("VVVV : \n");
    for (int i = 1; i <= V_MAX_IDX; i++) {
        printf("%d ", V[i]);
    }
    printf("\n");
    printf("EEE : \n");
    for (int i = 0; i <= 2 * V_MAX_IDX; i++) {
        printf("%d ", E[i]);
    }
    // printf("\n");
    // printf("WWW : \n");
    // for (int i = 0; i <= 2 * V_MAX_IDX; i++) {
    //     printf("%f ", W[i]);
    // }


    // TODO TODO TODO
    // zrobic zaleznie od numerowania


    // std::memset(V, '\0', sizeof(uint32_t) * V_MAX_IDX);
    // thrust::device_vector<uint32_t> GG(newID[newID.size() - 1] + 2, 0);
    // thrust::reduce_by_key(newV.begin(), newV.end(), newV.begin(), GG.begin() + 1, thrust::make_discard_iterator());
    // printf("\n");
    // printf("VVVV : \n");
    
    // thrust::copy(GG.begin(), GG.end(), 
    //         std::ostream_iterator<uint32_t>(std::cout, " "));

    // std::memset(V, '\0', sizeof(uint32_t) * V_MAX_IDX);
    // HANDLE_ERROR(cudaMemcpy((void*)V, (void*)RAW(GG), GG.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // uint32_t* tmpNewID = RAW(newID); // device lambda cannot capture by reference
    // thrust::transform(&E[0], &E[GG[GG.size() - 1]], &E[0], [=] __device__ (const uint32_t& x) {printf("tmp: %d\n", tmpNewID[x]); return tmpNewID[x];} );
    // for (int i = 1; i <= V_MAX_IDX; i++) {
    //     printf("%d ", V[i]);
    // }

    // debug_print(V_MAX_IDX, V, E, W, globalCommAssignment);


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

    HANDLE_ERROR(cudaHostGetDevicePointer(&V, std::get<1>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&E, std::get<2>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&W, std::get<3>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&k, std::get<4>(res), 0));

    uint32_t* tmp;
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(uint32_t) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&newComm, tmp, 0));
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(float) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&ac, tmp, 0));

    // communities separately, because they must be initialized
    // auto _comm = thrust::device_vector<uint32_t>(V_MAX_IDX + 1);
    // thrust::sequence(_comm.begin(), _comm.end());
    // comm = thrust::raw_pointer_cast(&_comm[0]);


    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(uint32_t) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    for (int i = 0; i <= V_MAX_IDX; i++) {
        tmp[i] = i; // each node is in it's own community at the beginning
    }
    HANDLE_ERROR(cudaHostGetDevicePointer(&comm, tmp, 0));

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte); // TODO customize
    // cudaFuncSetCacheConfig(reassign_nodes, cudaFuncCachePreferShared); // TODO

    cudaDeviceSynchronize();

    thrust::device_vector<uint32_t> globCommAssignment(V_MAX_IDX + 1);

    float mod = reassign_communities(V_MAX_IDX, V, E, W, k, ac, comm, newComm, m, MIN_GAIN, globCommAssignment);

    printf("TOTAL END MODularity: %f\n", mod);

    thrust::host_vector<uint32_t> resComm(V_MAX_IDX + 1);
    
    thrust::copy(globCommAssignment.begin(), globCommAssignment.end(), resComm.begin());


    if (VERBOSE) {
        // print_comm_assignment(V_MAX_IDX, comm);
        print_comm_assignment(V_MAX_IDX, RAW(resComm));
    }
    // TODO cleanup
    // cudaFree(V);
    return 0;
}
