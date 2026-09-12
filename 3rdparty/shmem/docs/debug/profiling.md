# 在样例工程使用Profiling工具进行性能采集

本小节介绍基于GetSystemCycle的Profiling工具进行分析数据搬运操作的耗时，通过采集系统时钟周期数并转换为实际时间，精准量化不同Block（计算核）、不同Frame（埋点 ID）下的MTE搬运性能。

Profiling工具核心特性：
按 Block（核）和 Frame（埋点 ID）维度统计操作执行次数（count）和总耗时周期（cycles）；
自动将周期数转换为微秒（us），精确到三位小数位输出；
仅显示有有效数据的Block和Frame，过滤空数据，提升可读性；

关于GetSystemCycle的详细介绍，可参考[GetSystemCycle](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0282.html)。

## 插入调试代码
1. 插入打点，修改kernel代码在需要打点的位置前后插入SHMEMI_PROF_START和SHMEMI_PROF_END，具体可参考`examples/allgather/allgather_kernel.cpp`。
    ```diff
    --- a/examples/allgather/allgather_kernel.cpp
    +++ b/examples/allgather/allgather_kernel.cpp
    @@ -12,6 +12,9 @@
    #include "acl/acl.h"
    #include "shmem.h"
    
    +// shmem prof
    +#include "utils/prof/shmemi_prof.h"
    +
    #undef inline
    #include "opdev/fp16_t.h"
    #include "opdev/bfloat16.h"
    @@ -72,6 +75,7 @@ ACLSHMEM_DEVICE void all_gather_origin(__gm__ T *input, __gm__ T *output, __gm__
            int64_t times = 0;
            int64_t flag = 0;
            while (copy_total_size >= copy_ub_size) {
    +            SHMEMI_PROF_START(0);
                aclshmemx_mte_put_nbi(gva_data_gm + aivIndex * len_per_core + times * copy_ub_num,
                                    input_gm + aivIndex * len_per_core + times * copy_ub_num, tmp_buff, copy_ub_size,
                                    copy_ub_num, my_rank, EVENT_ID0);
    @@ -80,7 +84,7 @@ ACLSHMEM_DEVICE void all_gather_origin(__gm__ T *input, __gm__ T *output, __gm__
                times += 1;
                flag = times + magic;
                aclshmemx_signal_op(gva_sync_gm + flag_offset, flag, ACLSHMEM_SIGNAL_SET, my_rank);
    -
    +            SHMEMI_PROF_END(0);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    ```

2. 在host侧的应用程序中，增加profiling数据展示，具体可参考`examples/allgather/main.cpp`：

    ```diff
    --- a/examples/allgather/main.cpp
    +++ b/examples/allgather/main.cpp
    @@ -120,6 +120,8 @@ int test_aclshmem_all_gather(int rank_id, int n_ranks)
            }
            status = aclrtSynchronizeStream(stream);

    +        aclshmemx_show_prof(nullptr, true);
    +
            // Result Check
            T *output_host;
            size_t output_size = n_ranks * trans_size * sizeof(T);
    ```
   注意：只打印采集的pe和block，无数据的会跳过打印。

## 编译运行

1. 按照如上步骤修改完后，进行编译算子样例。

   ```sh
   bash scripts/build.sh -examples
   ```
2. 在examples/allgather目录执行demo:

   ```sh
   export SHMEM_CYCLE_PROF_PE=0  # 设置需要采集的pe，当前仅支持采集某个指定的pe
   bash run.sh -ranks 2
   ```
3. 观察其demo打印，会有如下样式打印：
    ```sh
    ============================================================
    BlockID   FrameID   Cycles         Count          AvgTime(us)    
    ------------------------------------------------------------
    0         0         7506966        34050          4.409          
    1         0         7485800        34050          4.397          
    2         0         8290500        34050          4.870          
    3         0         8279083        34050          4.863          
    4         0         8255644        34050          4.849          
    5         0         8275272        34050          4.861          
    6         0         8429026        34050          4.951          
    7         0         8404425        34050          4.937          
    ============================================================
    ```
    字段描述

    | keyword      | description   |
    |--------------|---------------|
    | BlockID      | Device核的index   |
    | FrameID      | 埋点ID         |
    | Cycles       | 系统cycle数       |
    | Count        | 执行总次数        |
    | AvgTime(us)  | 执行平均时间      |