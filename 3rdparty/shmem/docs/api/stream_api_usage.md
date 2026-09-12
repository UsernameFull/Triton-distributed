# SHMEM C++ Stream API 使用说明

## 一、概述

本文档介绍四个基于Stream的SHMEM C++接口的使用方法，这些接口允许在指定的ACL流上执行数据传输和信号操作。

**跨机支持说明：**

| 接口                                         | 跨机支持 | 说明                                                                                                 |
| ------------------------------------------ | ---- | -------------------------------------------------------------------------------------------------- |
| aclshmemx\_putmem\_on\_stream              | 支持   | HCCS连通时优先走MTE，否则RDMA链路可用时走RDMA                                                                     |
| aclshmemx\_getmem\_on\_stream              | 支持   | HCCS连通时优先走MTE，否则RDMA链路可用时走RDMA                                                                     |
| aclshmemx\_signal\_op\_on\_stream          | 部分支持 | `ACLSHMEM_SIGNAL_SET`：HCCS连通时优先走MTE，否则RDMA链路可用时走RDMA；`ACLSHMEM_SIGNAL_ADD`不支持RDMA跨机，但HCCS可通时支持MTE跨机 |
| aclshmemx\_signal\_wait\_until\_on\_stream | 不支持  | 语义为等待本卡上地址到达某个值                                                                                    |

跨机器场景下的数据传输路径选择策略：

- **优先使用MTE接口**：当HCCS链路连通时，优先走MTE路径
- **RDMA接口**：当HCCS链路不通但RDMA配置可用时，自动切换到RDMA路径进行跨机通信

若要实现上述自动选择MTE/RDMA路径的功能，需要通过以下方式初始化：

```cpp
aclshmemx_init_attr_t attributes;
// 配置MTE和RDMA路径
attributes.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE);
int status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
```

完整初始化可参考`tests\unittest\host\main_test.cpp`的`void test_cross_init()`函数。

***

## 二、接口说明

### 1. aclshmemx\_putmem\_on\_stream

**功能描述：**

同步接口，在指定流上将对称内存中本卡上的数据从源地址复制到指定PE的地址上。

**跨机支持：** 支持，HCCS连通时优先走MTE，否则RDMA链路可用时走RDMA。

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmemx_putmem_on_stream(void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream);
```

**参数说明：**

| 参数         | 类型          | 说明                   |
| ---------- | ----------- | -------------------- |
| dst        | void\*      | 目标PE的目的地址在本卡的对称地址指针  |
| src        | void\*      | 本地源数据的指针             |
| elem\_size | size\_t     | 要传输的数据大小（字节）         |
| pe         | int32\_t    | 目标PE的编号              |
| stream     | aclrtStream | 使用的流（如果为NULL，则使用默认流） |

**使用示例：**
`tests\unittest\host\mem\shmem_host_put_stream_test.cpp`存在完整调用示例，展示了如何在指定流上使用`aclshmemx_putmem_on_stream`接口进行数据传输。下面仅展示核心流程。

```cpp
// 伪代码示例：aclshmemx_putmem_on_stream使用流程

// 1. 初始化（跨机场景需配置MTE和RDMA）
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. 创建流
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. 跨机调用RDMA接口时，必须使用aclshmem_malloc分配对称内存，不能使用aclrtMalloc
size_t data_size = 1024; // 1KB
int32_t target_pe = 1; // 目标PE编号
void* src_ptr = aclshmem_malloc(data_size);
void* dst_ptr = aclshmem_malloc(data_size);
// 4. src_ptr初始化赋值
// 5. 执行put操作
aclshmemx_putmem_on_stream(dst_ptr, src_ptr, data_size, target_pe, stream);

// 6. 跨机调用RDMA接口时，必须调用aclshmemx_handle_wait等待传输完成
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);
aclrtSynchronizeStream(stream);

// 7. 释放资源
aclshmem_free(src_ptr);
aclshmem_free(dst_ptr);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***

### 2. aclshmemx\_getmem\_on\_stream

**功能描述：**

同步接口，在指定流上将对称内存中指定PE上的连续数据复制到本地PE的地址上。

**跨机支持：** 支持。HCCS连通时优先走MTE，否则RDMA链路可用时走RDMA。

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmemx_getmem_on_stream(void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream);
```

**参数说明：**

| 参数         | 类型          | 说明                      |
| ---------- | ----------- | ----------------------- |
| dst        | void\*      | 本地PE上的指针（目标地址）          |
| src        | void\*      | 远端PE上的源数据地址在本卡的对称地址     |
| elem\_size | size\_t     | 要传输的数据大小（字节）            |
| pe         | int32\_t    | 远端PE的编号                 |
| stream     | aclrtStream | 使用的ACL流（如果为NULL，则使用默认流） |

**使用示例：**
`tests\unittest\host\mem\shmem_host_get_stream_test.cpp`存在完整调用示例，展示了如何在指定流上使用`aclshmemx_getmem_on_stream`接口进行数据传输。下面仅展示核心流程。

```cpp
// 伪代码示例：aclshmemx_getmem_on_stream使用流程

// 1. 初始化（跨机场景需配置MTE和RDMA）
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. 创建流
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. 跨机调用RDMA接口时，必须使用aclshmem_malloc分配对称内存，不能使用aclrtMalloc
size_t data_size = 1024; // 1KB
int32_t source_pe = 0; // 源PE编号
void* src_ptr = aclshmem_malloc(data_size);
void* dst_ptr = aclshmem_malloc(data_size);

// 4. 执行get操作
aclshmemx_getmem_on_stream(dst_ptr, src_ptr, data_size, source_pe, stream);

// 5. 跨机调用RDMA接口时，必须调用aclshmemx_handle_wait等待传输完成
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);

aclrtSynchronizeStream(stream);

// 6. 释放资源
aclshmem_free(src_ptr);
aclshmem_free(dst_ptr);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***

### 3. aclshmemx\_signal\_op\_on\_stream

**功能描述：**

同步接口，在指定的PE上对信号变量执行操作，该操作在给定的流上执行。通常与`aclshmemx_signal_wait_until_on_stream`配合使用。

**跨机支持：** 部分支持。

- `ACLSHMEM_SIGNAL_SET`：HCCS可通时支持MTE跨机，也支持RDMA跨机
- `ACLSHMEM_SIGNAL_ADD`：不支持RDMA跨机，HCCS可通时支持MTE跨机

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmemx_signal_op_on_stream(int32_t *sig_addr, int32_t signal, int sig_op, int pe, aclrtStream stream);
```

**参数说明：**

| 参数        | 类型          | 说明                                                                          |
| --------- | ----------- | --------------------------------------------------------------------------- |
| sig\_addr | int32\_t\*  | 信号变量在本卡上的对称地址                                                               |
| signal    | int32\_t    | 用于原子操作的值                                                                    |
| sig\_op   | int         | 在远程信号变量上执行的操作，支持：• `ACLSHMEM_SIGNAL_SET`：设置信号值• `ACLSHMEM_SIGNAL_ADD`：累加信号值 |
| pe        | int         | 远端PE的编号                                                                     |
| stream    | aclrtStream | 使用的ACL流（如果为NULL，则使用默认流）                                                     |

**使用示例：**
`tests\unittest\host\sync\signal\signal_host_test.cpp`存在完整调用示例，展示了如何在指定流上使用`aclshmemx_signal_op_on_stream`接口进行信号操作。下面仅展示核心流程。

```cpp
// 伪代码示例：aclshmemx_signal_op_on_stream使用流程

// 1. 初始化（跨机场景需配置MTE和RDMA）
aclshmemx_init_attr_t attributes;
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. 创建流
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. 跨机调用RDMA接口时，必须使用aclshmem_malloc分配对称内存，不能使用aclrtMalloc
int32_t target_pe = 1; // 目标PE编号
int32_t* signal_var = (int32_t*)aclshmem_malloc(sizeof(int32_t));

// 4. 设置信号值（支持RDMA跨机）
aclshmemx_signal_op_on_stream(signal_var, 1, ACLSHMEM_SIGNAL_SET, target_pe, stream);

// 跨机调用RDMA接口时，必须调用aclshmemx_handle_wait等待传输完成
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);
// 5. 累加信号值（不支持RDMA跨机，仅HCCS可通时支持MTE跨机）
aclshmemx_signal_op_on_stream(signal_var, 10, ACLSHMEM_SIGNAL_ADD, target_pe, stream);

// 6. 等待操作完成
aclrtSynchronizeStream(stream);

// 7. 释放资源
aclshmem_free(signal_var);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***

### 4. aclshmemx\_signal\_wait\_until\_on\_stream

**功能描述：**

同步接口，调用该接口会阻塞，直到调用PE上的sig\_addr值满足由比较操作符cmp和比较值cmp\_val指定的等待条件。

**跨机支持：** 不支持。

**函数原型：**

```cpp
ACLSHMEM_HOST_API void aclshmemx_signal_wait_until_on_stream(int32_t *sig_addr, int cmp, int32_t cmp_val, aclrtStream stream);
```

**参数说明：**

| 参数        | 类型          | 说明                                                                                                                                                 |
| --------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| sig\_addr | int32\_t\*  | 源信号变量的本地地址                                                                                                                                         |
| cmp       | int         | 比较操作符，支持：• `ACLSHMEM_CMP_EQ`：等于• `ACLSHMEM_CMP_NE`：不等于• `ACLSHMEM_CMP_GT`：大于• `ACLSHMEM_CMP_GE`：大于等于• `ACLSHMEM_CMP_LT`：小于• `ACLSHMEM_CMP_LE`：小于等于 |
| cmp\_val  | int32\_t    | 用于比较的数值                                                                                                                                            |
| stream    | aclrtStream | 使用的ACL流（如果为NULL，则使用默认流）                                                                                                                            |

**使用示例：**
`tests\unittest\host\sync\signal\signal_host_test.cpp`存在完整调用示例，展示了如何在指定流上使用`aclshmemx_signal_wait_until_on_stream`接口等待信号条件。下面仅展示核心流程。

```cpp
// 伪代码示例：aclshmemx_signal_wait_until_on_stream使用流程

// 1. 初始化
aclshmemx_init_attr_t attributes;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. 创建流
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. 跨机调用RDMA接口时，必须使用aclshmem_malloc分配对称内存，不能使用aclrtMalloc
int32_t* signal_var = (int32_t*)aclshmem_malloc(sizeof(int32_t));

// 4. 设置信号值
aclshmemx_signal_op_on_stream(signal_var, 2, ACLSHMEM_SIGNAL_SET, target_pe, stream);

// 4. 等待本地信号等于指定值（阻塞直到条件满足），通常为多卡场景
aclshmemx_signal_wait_until_on_stream(signal_var, ACLSHMEM_CMP_EQ, 2, stream);

// 6. 等待操作完成
aclrtSynchronizeStream(stream);

// 7. 释放资源
aclshmem_free(signal_var);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***

### 典型使用场景：环形信号同步

以下示例展示多个PE之间的环形信号同步场景：每个PE等待上一个PE设置的信号，然后设置下一个PE的信号。完整示例见`tests\unittest\host\sync\signal\signal_host_test.cpp`的`test_signal_eq_all_pes`函数。

```cpp
// 伪代码示例：环形信号同步场景
// 假设有N个PE，PE0 -> PE1 -> PE2 -> ... -> PE(N-1) -> PE0 形成环形

// 1. 初始化
aclshmemx_init_attr_t attributes;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// 2. 创建流
aclrtStream stream;
aclrtCreateStream(&stream);

// 3. 分配信号变量（对称内存）
int32_t* sig_addr = (int32_t*)aclshmem_malloc(sizeof(int32_t));
aclrtMemset(sig_addr, sizeof(int32_t), 0, sizeof(int32_t)); // 初始化为0

// 4. 计算环形关系
int32_t expected_signal = pe_id + 1;        // 期望收到的信号值（来自上一个PE）
int next_pe = (pe_id + 1) % n_pes;          // 下一个PE编号
int32_t my_signal = next_pe + 1;            // 要发送给下一个PE的信号值

// 5. 设置下一个PE的信号值
aclshmemx_signal_op_on_stream(sig_addr, my_signal, ACLSHMEM_SIGNAL_SET, next_pe, stream);

// 调用aclshmemx_handle_wait等待信号设置完成
aclshmem_handle_t handle;
handle.team_id = ACLSHMEM_TEAM_WORLD;
aclshmemx_handle_wait(handle, stream);

aclrtSynchronizeStream(stream);

// 6. 等待当前PE收到期望的信号值（阻塞直到条件满足）
aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_EQ, expected_signal, stream);
aclrtSynchronizeStream(stream);

// 7. 验证信号值
int32_t host_value;
aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
// host_value 应该等于 expected_signal

// 8. 释放资源
aclshmem_free(sig_addr);
aclshmem_finalize();
aclrtDestroyStream(stream);
aclrtResetDevice(device_id);
aclFinalize();
```

***
## 三、注意事项

1. **内存申请**：跨机调用RDMA接口时，必须使用`aclshmem_malloc`分配对称内存。
2. **跨机通信**：
   - `aclshmemx_putmem_on_stream`和`aclshmemx_getmem_on_stream`支持跨机，HCCS连通时优先走MTE，否则RDMA链路可用时走RDMA
   - `aclshmemx_signal_op_on_stream`：`ACLSHMEM_SIGNAL_SET`支持RDMA跨机；`ACLSHMEM_SIGNAL_ADD`不支持RDMA跨机，HCCS可通时支持MTE跨机
   - `aclshmemx_signal_wait_until_on_stream`不支持RDMA跨机，HCCS可通时支持MTE跨机
   - 跨机场景下需确保网络配置正确，相关环境变量已正确设置
   - **RDMA路径同步**：当使用RDMA接口时，需要调用`aclshmemx_handle_wait`来保证数据已经被接收完成。

