# SHMEM API 样例
SHMEM包含host和device两类接口。host接口用ACLSHMEM_HOST_API宏标识，device接口用ACLSHMEM_DEVICE宏标识。

此章介绍各类API的常见接口的使用样例。

## Init API
SHMEM的初始化接口样例

### 初始化状态
```c++
enum {
    ACLSHMEM_STATUS_NOT_INITIALIZED = 0,    // 未初始化
    ACLSHMEM_STATUS_SHM_CREATED,           // 完成共享内存堆创建
    ACLSHMEM_STATUS_IS_INITIALIZED,         // 初始化完成
    ACLSHMEM_STATUS_INVALID = INT_MAX,
};
```

### 初始化所需的attributes
```c++
// 初始化属性
typedef struct {
    int version;                            // 版本
    int my_rank;                             // 当前rank
    int n_ranks;                             // 总rank数
    char ip_port[ACLSHMEM_MAX_IP_PORT_LEN];      // ip端口
    uint64_t local_mem_size;                  // 本地申请内存大小
    aclshmem_init_optional_attr_t option_attr;  // 可选参数
} aclshmemx_init_attr_t;

// 可选属性
typedef struct {
    data_op_engine_type_t data_op_engine_type; // 数据引擎
    // timeout
    uint32_t shm_init_timeout;
    uint32_t shm_create_timeout;
    uint32_t control_operation_timeout;
    // apply avaliable port in advance
    int32_t sockFd;
} aclshmem_init_optional_attr_t;
```

### 初始化样例
```c++
#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "shmem.h"
aclInit(nullptr);
int status;
int device_id = 0;
int rank_id = 0;
int n_ranks = 8;
uint64_t local_mem_size = 1024UL * 1024UL * 1024;
const char* test_global_ipport = "tcp://127.0.0.1:8666";
status = aclrtSetDevice(device_id);

aclshmemx_init_attr_t* attributes = new aclshmemx_init_attr_t{rank_id, n_ranks, test_global_ipport, local_mem_size, {0, ACLSHMEM_DATA_OP_MTE, 120, 120, 120}}; // 自定义attr
aclshmemx_init_attr(attributes);
delete attributes;
attributes = nullptr;

status = aclshmemx_init_status();
if (status == ACLSHMEM_STATUS_IS_INITIALIZED) {
    std::cout << "Init success!" << std::endl;
}
//################你的任务#################

//#########################################
status = aclshmem_finalize();
aclrtResetDevice(device_id);
aclFinalize();

```

### 自定义日志打印

自定义日志打印是可选操作，如果未注册自定义日志打印函数，日志则默认打屏，接口示例如下。
```c
void cpp_logger_example(int level, const char* msg)
{
    // do print here
}

// set self-defined log
int ret = aclshmemx_set_extern_logger(cpp_logger_example);

// set log level. 0-debug, 1-info, 2-warn, 3-error
ret = aclshmemx_set_log_level(level);
```

### 注册私钥口令解密函数

shmem的多卡之间业务面TCP通信，提供的能力实现，为了保证通信安全，该特性默认开启，使用时传入初始化TLS信息，TLS信息格式参考SECURITY.md样例，其中私钥是密文，私钥口令以密文形式存储，需通过注册的解密回调函数进行解密，调用者实现具体的解密逻辑。

```c
int my_key_password_decrypt_handler(const char *cipherText, size_t cipherTextLen, char *plainText, size_t &plainTextLen)
{
    // cipherText: input encrypted key password
    // plainText: output decrypted key password
    // plainTextLen: output decrypted key password length
    // do decrypt here
}

const char *pk = "xxx";
uint32_t pk_len = strlen(pk);

const char *password = "xxxx";
uint32_t pw_len = strlen(password);
int ret = aclshmemx_set_config_store_tls_key(pk, pk_len, password, pw_len, my_key_password_decrypt_handler);
```

如需关闭加密特性，则调用如下接口。关闭后，则无需调用aclshmem_set_config_store_tls_key接口。
```c
int ret = aclshmemx_set_conf_store_tls(false, nullptr, 0);
```

### Team API
SHMEM的通信域管理接口样例

### host侧接口样例

```c++
// ################### 调用初始化相关接口 ###########################
//...
// ###################### 子通信域切分 #############################
aclshmem_team_t team_odd;
int start = 1;
int stride = 2;
int team_size = 4;
aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

// ##################### host侧取值 ###############################
if (rank_id & 1) {

    // aclshmem_team_n_pes(team_odd): Returns the number of PEs in the team.
    int team_n_pes = aclshmem_team_n_pes(team_odd); // team_n_pes == team_size
    // aclshmem_team_my_pe(team_odd): Returns the number of the calling PE in the specified team.
    int team_my_pe = aclshmem_team_my_pe(team_odd); // team_my_pe == rank_id / stride
    // aclshmem_n_pes(): Returns the number of PEs running in the program.
    int n_pes = aclshmem_n_pes(); // n_pes == n_ranks
    // aclshmem_my_pe(): Returns the PE number of the local PE
    int my_pe = aclshmem_my_pe(); // my_pe == rank_id
}

// #################### 相关资源释放 ################################
aclshmem_team_destroy(team_odd);
// ################## 调用去初始化相关接口 ###########################
//...
```

### device侧接口样例
```c++
class kernel_state_test {
public:
    __aicore__ inline kernel_state_test() {}
    __aicore__ inline void Init(GM_ADDR gva, aclshmem_team_t team_id)
    {
        gva_gm = (__gm__ int *)gva;
        team_idx= team_id;

        rank = aclshmem_my_pe();          // 获取当前rank
        rank_size = aclshmem_n_pes(); // 获取总rank数
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        // ##################### device侧取值 ###############################
        // aclshmem_int32_p 是RMA功能提供的接口，此处可简易理解为在device存储第二个入参的函数的结果。

        // aclshmem_n_pes(): Returns the number of PEs running in the program.
        aclshmem_int32_p(gva_gm, aclshmem_n_pes(), rank);
        // aclshmem_my_pe(): Returns the PE number of the local PE
        aclshmem_int32_p(gva_gm + 1, aclshmem_my_pe(), rank);
        // aclshmem_team_my_pe(team_idx): Returns the number of the calling PE in the specified team.
        aclshmem_int32_p(gva_gm + 2, aclshmem_team_my_pe(team_idx), rank);
        // aclshmem_team_n_pes(team_idx): Returns the number of PEs in the team.
        aclshmem_int32_p(gva_gm + 3, aclshmem_team_n_pes(team_idx), rank);
        // aclshmem_team_translate_pe(team_idx, 1, ACLSHMEM_TEAM_WORLD): Translate a given PE number in one team into the corresponding PE number in another team.
        aclshmem_int32_p(gva_gm + 4, aclshmem_team_translate_pe(team_idx, 1, ACLSHMEM_TEAM_WORLD), rank);
    }
private:
    __gm__ int *gva_gm;
    aclshmem_team_t team_idx;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void device_state_test(GM_ADDR gva, int team_id)
{
    kernel_state_test op;
    op.Init(gva, (aclshmem_team_t)team_id);
    op.Process();
}

void get_device_state(uint32_t block_dim, void* stream, uint8_t* gva, aclshmem_team_t team_id)
{
    device_state_test<<<block_dim, nullptr, stream>>>(gva, (int)team_id);
}
```
## Mem API
SHMEM的内存管理接口样例

```c++
// ################### 调用初始化相关接口 ###########################
//...
// ################## 内存管理接口调用 ###########################
// 分配1024 bytes，返回所分配内存的指针ptr.
void *ptr = aclshmem_malloc(1024);
// 释放ptr对应的被分配的内存空间.
aclshmem_free(ptr);
// ################## 调用去初始化相关接口 ###########################
//...
```

## RMA API
SHMEM的远端内存访问接口样例

```c++
class kernel_p {
public:
    __aicore__ inline kernel_p() {}
    __aicore__ inline void Init(GM_ADDR gva, float val)
    {
        gva_gm = (__gm__ float *)gva;
        value = val;

        rank = aclshmem_my_pe();          // 获取当前rank
        rank_size = aclshmem_n_pes(); // 获取总rank数
    }
    __aicore__ inline void Process()
    {
        // 把value的值put到共享内存gva_gm在(rank + 1) % rank_size中的对应位置。
        aclshmem_float_p(gva_gm, value, (rank + 1) % rank_size);
    }
private:
    __gm__ float *gva_gm;
    float value;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void p_num_test(GM_ADDR gva, float val)
{
    kernel_p op;
    op.Init(gva, val);
    op.Process();
}

void put_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, float val)
{
    p_num_test<<<block_dim, nullptr, stream>>>(gva, val);
}
```

## Sync API
SHMEM的同步管理接口样例

```c++
// 任务1
// ...
// 阻塞直到所有任务完成。
aclshmem_barrier_all();
// 任务2
// ...
```
