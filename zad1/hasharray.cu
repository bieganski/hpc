#include <cassert>
#include <stdio.h>

#include "hasharray.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

typedef HashArray HA;

static const uint32_t kEmpty = 0xffffffff;

static void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))


CUDA_CALLABLE_MEMBER uint32_t hash(uint32_t k, uint32_t table_size) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (table_size - 1);
}

__host__ void HA::init(KeyValue* memory, uint32_t num) {
    assert(kEmpty == 0xffffffff);
    assert((num & kEmpty) == num); // power of 2
    HANDLE_ERROR(cudaMemset(memory, 0xff, sizeof(KeyValue) * num));
}

// CUDA_CALLABLE_MEMBER uint32_t HA::insert(KeyValue* hashtable, uint32_t key, uint32_t value, uint32_t table_size) {
//     uint32_t slot = hash(key, table_size);

//     while (true) {
//         uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
//         // printf("%d,", prev);
//         // return prev;
//         if (prev == kEmpty || prev == key) {
//             hashtable[slot].value = value;
//             return slot;
//         }
//         slot = (slot + 1) & (table_size - 1);
//     }
// }

CUDA_CALLABLE_MEMBER uint32_t HA::addInt(KeyValueInt* hashtable, uint32_t key, uint32_t value, uint32_t table_size) {
    uint32_t slot = hash(key, table_size);

    while (true) {
        uint32_t prev_key = atomicCAS(&hashtable[slot].key, kEmpty, key);
        
        if (prev_key == kEmpty) {
            uint32_t prev_val = atomicCAS(&hashtable[slot].value, kEmpty, value);
            if (prev_val != kEmpty) {
                atomicAdd(&hashtable[slot].value, value);
            }
            return slot;
        } else if (prev_key == key) {
            atomicAdd(&hashtable[slot].value, value);
            return slot;
        }
        slot = (slot + 1) & (table_size - 1);
    }
}



CUDA_CALLABLE_MEMBER uint32_t HA::lookupInt(KeyValueInt* hashtable, uint32_t key, uint32_t table_size) {
    uint32_t slot = hash(key, table_size);

    while (true) {
        if (hashtable[slot].key == key)
        {
            return hashtable[slot].value;
        }
        if (hashtable[slot].key == kEmpty)
        {
            return kEmpty;
        }
        slot = (slot + 1) & (table_size - 1);
    }
}


CUDA_CALLABLE_MEMBER uint32_t HA::addFloat(KeyValueFloat* hashtable, uint32_t key, float value, uint32_t table_size) {
    uint32_t slot = hash(key, table_size);

    while (true) {
        uint32_t prev_key = atomicCAS(&hashtable[slot].key, kEmpty, key);
        
        if (prev_key == kEmpty) {
            float prev_val = atomicCAS(&hashtable[slot].value, kEmpty, value);
            if (prev_val != (float) kEmpty) {
                atomicAdd(&hashtable[slot].value, value);
            }
            return slot;
        } else if (prev_key == key) {
            atomicAdd(&hashtable[slot].value, value);
            return slot;
        }
        slot = (slot + 1) & (table_size - 1);
    }
}



CUDA_CALLABLE_MEMBER float HA::lookupFloat(KeyValueFloat* hashtable, uint32_t key, uint32_t table_size) {
    uint32_t slot = hash(key, table_size);

    while (true) {
        if (hashtable[slot].key == key)
        {
            return hashtable[slot].value;
        }
        if (hashtable[slot].key == kEmpty)
        {
            return kEmpty;
        }
        slot = (slot + 1) & (table_size - 1);
    }
}