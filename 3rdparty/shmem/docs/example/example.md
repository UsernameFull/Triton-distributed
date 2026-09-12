# 初始化
该样例工程位于`examples/init`文件夹下。

SHMEM当前提供了三种flag用于初始化：
    - ACLSHMEMX_INIT_WITH_DEFAULT 默认初始化流程，可以不依赖第三方库完成初始化，但需要保证设定的ipport空闲
    - ACLSHMEMX_INIT_WITH_MPI 依赖MPI的多进程管理能力的初始化流程
    - ACLSHMEMX_INIT_WITH_UNIQUEID 运行时依赖第三方库广播能力（MPI_Bcast/torch.distributed.broadcast等）的初始化流程

在该样例中，分别实现了SHMEM提供的所有初始化流程调用方式。可以通过脚本参数编译运行不同flag对应的初始化流程。默认使用回环地址127.0.0.1，端口8666。该用例仅用以单机示范，请保证端口空闲可绑定。

## ACLSHMEMX_INIT_WITH_DEFAULT

### 样例运行
在根目录调用`scripts/build.sh`控制后端。
```bash
bash scripts/build.sh
```

```bash
cd examples/init
# 编译并运行两个pe的default初始化流程
bash run.sh -mode default -pesize 2
```
### 代码实现
SHMEM初始化接口调用前需要完成attributes创建，此用例提供了一种attributes构建方式，也可根据实际需要参考`aclshmemx_init_attr_t`定义自行设置。
```cpp
// attributes设置函数
int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                       aclshmemx_uniqueid_t *default_flag_uid, aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), (size_t)(ACLSHMEM_MAX_IP_PORT_LEN - 1));

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(default_flag_uid);
    aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);

    return ACLSHMEM_SUCCESS;
}
```

```cpp
//default初始化流程
int run_main(int argc, char* argv[]) {
    int pe = atoi(argv[1]);
    int pe_size = atoi(argv[2]);
    std::string ipport = argv[3];
    aclshmemx_uniqueid_t default_flag_uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    aclshmemx_init_attr_t attributes;

    int status = ACLSHMEM_SUCCESS;
    // acl初始化
    aclInit(nullptr);
    aclrtSetDevice(pe);
    
    uint64_t local_mem_size = 1024 * 1024 * 1024;
    // 创建attributes
    test_set_attr(pe, pe_size, local_mem_size, ipport.c_str(), &default_flag_uid, &attributes);
    // 初始化
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    /*
    你的任务
    */

    //去初始化
    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();

    return 0;
}
```
## ACLSHMEMX_INIT_WITH_MPI

### 样例运行
编译前需要安装并配置MPI环境变量，不配置无法编译MPI能力so。
```bash
# mpich安装在默认路径时可参考，需根据实际情况替换。
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```

在根目录调用`scripts/build.sh`
```bash
bash scripts/build.sh
```

```bash
cd examples/init
# 编译并运行两个pe的mpi初始化流程
bash run.sh -mode mpi -pesize 2
```
### 代码实现
```cpp
//mpi初始化流程
int run_main() {
    // MPI初始化
    MPI_Init(nullptr, nullptr);
    int pe;
    int pe_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);
    int status = ACLSHMEM_SUCCESS;
    // acl初始化
    aclInit(nullptr);
    aclrtSetDevice(pe);
    uint64_t local_mem_size = 1024 * 1024 * 1024;
    // 创建attributes，依赖MPI无需使用ipport可以不传。
    aclshmemx_init_attr_t attributes = {
        pe, pe_size, "", local_mem_size, {0, ACLSHMEM_DATA_OP_MTE, 120, 120, 120}};
    // 初始化
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_MPI, &attributes);

    /*
    你的任务
    */

    //去初始化
    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();
    MPI_Finalize();
    return 0;
}
```
## ACLSHMEMX_INIT_WITH_UNIQUEID
### 样例运行
编译前需要安装并配置MPI环境变量，不配置无法编译MPI能力so。
```bash
# mpich安装在默认路径时可参考，需根据实际情况替换。
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```

在根目录调用`scripts/build.sh`控制后端。
```bash
bash scripts/build.sh
```

```bash
cd examples/init
# 编译并运行两个pe的mpi初始化流程
bash run.sh -mode mpi -pesize 2
```
### 代码实现
```cpp
//uid初始化流程
int run_main() {
    // MPI初始化
    MPI_Init(nullptr, nullptr);
    int pe;
    int pe_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);
    int status = ACLSHMEM_SUCCESS;
    // acl初始化
    aclInit(nullptr);
    aclrtSetDevice(pe);
    
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    
    int64_t local_mem_size = 1024 * 1024 * 1024;
    // pe 0 获取uid
    if (pe == 0) {
        status = aclshmemx_get_uniqueid(&uid);
    }
    // 广播pe 0 uid到其他pe
    MPI_Bcast(&uid, sizeof(aclshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
    // 调用uid的set接口创建attributes。
    status = aclshmemx_set_attr_uniqueid_args(pe, pe_size, 
                                                local_mem_size, 
                                                &uid, &attributes);
    // 初始化                                            
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);

    /*
    你的任务
    */

    //去初始化
    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();
    MPI_Finalize();
    return 0;

}
```
# AllGather
该样例工程位于`examples/allgather`文件夹下。

在该样例中，实现了一个通信量较小(单PE通信量小于2MB)情况下，有着更低时延的AllGather纯通信算子。 各PE首先将存在本端input地址下的数据push到本PE的对称内存上；确认远端PE的任务完成后，从远端PE的对称内存拉取对应PE的数据，从而整体完成AllGather的操作。这个样例展示了多种shmem API的用法，包括aclshmemx_mte_put_nbi、aclshmemx_signal_op以及aclshmemx_mte_get_nbi等，用于p2p的通信以及同步任务。

## 核函数实现
```c++

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"
using namespace AscendC;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

template<typename T>
ACLSHMEM_DEVICE void all_gather_small_data(uint64_t fftsAddr, __gm__ T* input, __gm__ T* output, __gm__ T* gva, int elements, int magic)
{
    const int64_t aivNum = GetBlockNum() * 2;
    const int64_t aivIndex = GetBlockIdx();

    const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
    const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    __gm__ T *input_gm = (__gm__ T *)input;
    __gm__ T *output_gm = (__gm__ T *)output;

    __gm__ T *gva_data_gm = (__gm__ T*)((__gm__ int32_t*)gva + data_offset);
    __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;
    
    __ubuf__ T* tmp_buff = (__ubuf__ T*)(64);

    // data move parameters
    const uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t input_offset, output_offset, gva_offset, num_per_core;

    // [AllGather Step 1] local input gm -> symmetric mem.
    num_per_core = elements / aivNum;
    input_offset = aivIndex * num_per_core;
    gva_offset = aivIndex * num_per_core;
    if (aivIndex == aivNum - 1) {
        num_per_core = elements - num_per_core * aivIndex;
    }
    aclshmemx_mte_put_nbi(gva_data_gm + gva_offset, input_gm + input_offset, tmp_buff, ub_size, num_per_core, my_rank, EVENT_ID0);

    const int64_t core_per_rank = aivNum / pe_size;
    const int64_t core_rank_idx = aivIndex % core_per_rank;
    const int64_t x = aivIndex / core_per_rank;

    // Sync Ensure Corresponding Tasks Done.
    aclshmem_quiet();

    aclshmemx_signal_op(gva_sync_gm + flag_offset, magic, ACLSHMEM_SIGNAL_SET, my_rank);
    aclshmem_signal_wait_until((__gm__ int32_t *)aclshmem_ptr(gva_sync_gm, x) + flag_offset, ACLSHMEM_CMP_EQ, magic);

    // [AllGather Step 2] symmetric mem -> local output.
    num_per_core = elements / core_per_rank;
    output_offset = x * elements + core_rank_idx * num_per_core;
    gva_offset = core_rank_idx * num_per_core;
    if (core_rank_idx == core_per_rank - 1) {
        num_per_core = elements - num_per_core * core_rank_idx;
    }
    aclshmemx_mte_get_nbi(output_gm + output_offset, gva_data_gm + gva_offset, tmp_buff, ub_size, num_per_core, x, EVENT_ID0);
}
```