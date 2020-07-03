#include <cassert>
#include <stdio.h>

#include "hasharray.h"
#include "utils.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

typedef HashArray HA;


CUDA_CALLABLE_MEMBER uint32_t hash(uint32_t k, uint32_t table_size) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (table_size - 1);
}

__host__ void HA::init(KeyValueInt* memory, uint32_t num) {
    assert(false);
    // assert(hashArrayNull == 0xffffffff);
    // assert((num & hashArrayNull) == num); // power of 2
    // HANDLE_ERROR(cudaMemset(memory, 0xff, sizeof(KeyValueInt) * num));
}

__host__ void HA::init(KeyValueFloat* memory, uint32_t num) {
    assert(false);
    // assert(hashArrayNull == 0xffffffff);
    // assert((num & hashArrayNull) == num); // power of 2
    // HANDLE_ERROR(cudaMemset(memory, 0xff, sizeof(KeyValueFloat) * num));
}

CUDA_CALLABLE_MEMBER uint32_t HA::insertInt(KeyValueInt* hashtable, uint32_t key, uint32_t value, uint32_t table_size) {
    uint32_t slot = hash(key, table_size);

    while (true) {
        uint32_t prev = atomicCAS(&hashtable[slot].key, hashArrayNull, key);
        if (prev == hashArrayNull || prev == key) {
            hashtable[slot].value = value; // no need to lock, at least one write will be successful
            return slot;
        }
        slot = (slot + 1) & (table_size - 1);
    }
}


CUDA_CALLABLE_MEMBER bool HA::insertWithFeedback(KeyValueInt* h1, KeyValueFloat* h2, uint32_t key, uint32_t v1, float v2, uint32_t table_size) {
    uint32_t slot = hash(key, table_size);

    assert(next_2_pow(table_size) == table_size);

    while (true) {
        uint32_t prev = atomicCAS(&h1[slot].key, hashArrayNull, key);
        if (prev == hashArrayNull) {
            h1[slot].value = v1;
            atomicAdd(&h2[slot].value, v2);
            return true;
        } else if (prev == key) {
            atomicAdd(&h2[slot].value, v2);
            if (h1[slot].value != v1) {
                printf("UWAGA: chcę %d a był %d\n", v1, h1[slot].value);
            }
            // assert(h1[slot].value == v1);
            return false;
        }
        slot = (slot + 1) & (table_size - 1);
    }
}

// CUDA_CALLABLE_MEMBER uint32_t HA::addInt(KeyValueInt* hashtable, uint32_t key, uint32_t value, uint32_t table_size) {
//     uint32_t slot = hash(key, table_size);

//     while (true) {
//         uint32_t prev_key = atomicCAS(&hashtable[slot].key, hashArrayNull, key);
        
//         if (prev_key == hashArrayNull) {
//             uint32_t prev_val = atomicCAS(&hashtable[slot].value, hashArrayNull, value);
//             if (prev_val != hashArrayNull) {
//                 atomicAdd(&hashtable[slot].value, value);
//             }
//             return slot;
//         } else if (prev_key == key) {
//             atomicAdd(&hashtable[slot].value, value);
//             return slot;
//         }
//         slot = (slot + 1) & (table_size - 1);
//     }
// }



CUDA_CALLABLE_MEMBER uint32_t HA::lookupInt(KeyValueInt* hashtable, uint32_t key, uint32_t table_size) {
    uint32_t slot = hash(key, table_size);

    while (true) {
        if (hashtable[slot].key == key)
        {
            return hashtable[slot].value;
        }
        if (hashtable[slot].key == hashArrayNull)
        {
            assert(false);
            return hashArrayNull;
        }
        slot = (slot + 1) & (table_size - 1);
    }
}


// CUDA_CALLABLE_MEMBER uint32_t HA::addFloat(KeyValueFloat* hashtable, uint32_t key, float value, uint32_t table_size) {
//     uint32_t slot = hash(key, table_size);

//     while (true) {
//         uint32_t prev_key = atomicCAS(&hashtable[slot].key, hashArrayNull, key);
        
//         if (prev_key == hashArrayNull) {
//             uint32_t prev_val = atomicCAS((uint32_t*) &hashtable[slot].value, hashArrayNull, (uint32_t) value);
//             if (prev_val != hashArrayNull) {
//                 atomicAdd(&hashtable[slot].value, value);
//             }
//             return slot;
//         } else if (prev_key == key) {
//             atomicAdd(&hashtable[slot].value, value);
//             return slot;
//         }
//         slot = (slot + 1) & (table_size - 1);
//     }
// }

// return new value
CUDA_CALLABLE_MEMBER float HA::addFloatSpecificPos(KeyValueFloat* hashtable, uint32_t slot, float addend) {
    return atomicAdd(&hashtable[slot].value, addend);
}



CUDA_CALLABLE_MEMBER float HA::lookupFloat(KeyValueFloat* hashtable, uint32_t key, uint32_t table_size) {
    uint32_t slot = hash(key, table_size);

    while (true) {
        if (hashtable[slot].key == key)
        {
            return hashtable[slot].value;
        }
        if (hashtable[slot].key == hashArrayNull)
        {
            assert(false);
            return hashArrayNull;
        }
        slot = (slot + 1) & (table_size - 1);
    }
}