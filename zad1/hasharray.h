#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cstdint>

const uint32_t hashArrayNull = 0xFFFFFFFF;

typedef struct KeyValueFloat {
    uint32_t key;
    float value;
} KeyValueFloat;

typedef struct KeyValueInt {
    uint32_t key;
    uint32_t value;
} KeyValueInt;

class HashArray {
public:
    /**
     * `num` must be power of 2, because of slowness of modulo operator.
     */
    __host__ static void init(KeyValueInt* memory, uint32_t num);

    __host__ static void init(KeyValueFloat* memory, uint32_t num);

    // returns key
    CUDA_CALLABLE_MEMBER static uint32_t insertInt(KeyValueInt* hashtable, uint32_t key, uint32_t value, uint32_t table_size);

    // // returns key
    // CUDA_CALLABLE_MEMBER static uint32_t addInt(KeyValueInt* hashtable, uint32_t key, uint32_t value, uint32_t table_size);

    // returns value
    CUDA_CALLABLE_MEMBER static uint32_t lookupInt(KeyValueInt* hashtable, uint32_t key, uint32_t table_size);

    // // returns key
    // CUDA_CALLABLE_MEMBER static uint32_t addFloat(KeyValueFloat* hashtable, uint32_t key, float value, uint32_t table_size);

    // return new value
    CUDA_CALLABLE_MEMBER static float addFloatSpecificPos(KeyValueFloat* hashtable, uint32_t slot, float addend);

    // returns value
    CUDA_CALLABLE_MEMBER static float lookupFloat(KeyValueFloat* hashtable, uint32_t key, uint32_t table_size);

    /**
     * Finds for place for `key` in h1 (which is isomorphic with h2), when found (empty or occupied by same key),
     * it atomic adds v2 to h2 table at same idx.
     * Returns true if occupied new slot, false otherwise.
     */ 
    CUDA_CALLABLE_MEMBER static bool insertWithFeedback(KeyValueInt* h1, KeyValueFloat* h2, uint32_t key, uint32_t v1, float v2, uint32_t table_size);

    // static void delete_key(KeyValue* hashtable, uint32_t key)
};