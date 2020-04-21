#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cstdint>

typedef struct KeyValue
{
    uint32_t key;
    uint32_t value;
} KeyValue;

class HashArray {
public:
    /**
     * `num` must be power of 2, because of slowness of modulo operator.
     */
    __host__ static void init(KeyValue* memory, uint32_t num);

    // returns key
    CUDA_CALLABLE_MEMBER static uint32_t insert(KeyValue* hashtable, uint32_t key, uint32_t value, uint32_t table_size);

    // returns key
    CUDA_CALLABLE_MEMBER static uint32_t add(KeyValue* hashtable, uint32_t key, uint32_t value, uint32_t table_size);

    // returns value
    CUDA_CALLABLE_MEMBER static uint32_t lookup(KeyValue* hashtable, uint32_t key, uint32_t table_size);

    // static void delete_key(KeyValue* hashtable, uint32_t key)
};