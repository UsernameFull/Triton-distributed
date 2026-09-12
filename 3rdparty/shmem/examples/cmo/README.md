# CMO (Cache Maintenance Operation) 功能演示与读性能测试示例

## 功能简介

本示例演示了如何使用Shmem的CMO（Cache Maintenance Operation）接口来优化GM（Global Memory）内存访问性能。该CMO接口提供L2缓存管理操作，可以通过预取（Prefetch）将数据从GM提前加载到L2缓存中，从而减少数据访问延迟，提升整体计算性能。

### L2缓存背景知识

昇腾AI处理器采用多级缓存架构，L2缓存是二级缓存，位于AI Core和全局内存（HBM）之间，具有以下特点：

- **容量**：大容量高速缓存（A2/A3 经典值192MB）
- **访问速度**：缓存命中带宽约为缓存miss带宽的2~4倍
- **缓存管理**：支持提前将数据加载到缓存，掩盖内存访问延迟

通过合理使用CMO预取操作，可以在计算进行的同时提前准备下一批数据，提升整体性能。

### 测试场景

一、本示例对比了以下三种缓存预取操作策略下，GM读的性能表现：

1. **无预取 (NO_PREFETCH)**：直接从GM拷贝数据，不使用任何缓存优化
2. **Host侧预取 (HOST_PREFETCH)**：使用Host侧接口`aclrtCmoAsync`对整块需要拷贝的位置进行预取
3. **Device块内预取 (DEVICE_BLOCK_PREFETCH)**：在kernel内部分别对各个块需要拷贝的内存位置进行CMO预取操作

二、测试Device侧CMO接口`aclshmemx_cmo_nbi`的性能表现，对比了不同预取大小下的操作时延。

### 核心接口

#### CMO接口（SHMEM扩展接口）

```c
template <typename T>
void aclshmemx_cmo_nbi(__gm__ T *src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type,
                     __ubuf__ T *buf, uint32_t ub_size, uint32_t sync_id);
```

- **功能**：在Device侧异步触发CMO操作，向STARS队列提交操作任务
- **参数说明**：
  - `src`：全局内存地址
  - `elem_size`：元素数量
  - `cmo_type`：CMO操作类型（当前仅支持CMO_TYPE_PREFETCH）
  - `buf`：临时UB缓冲区地址
  - `ub_size`：UB缓冲区大小（至少64字节，64字节对齐）
  - `sync_id`：同步ID
- **特点**：基于SDMA引擎实现，支持核级细粒度控制

##### CMO操作类型

**注意**：当前SHMEM实现仅支持`CMO_TYPE_PREFETCH`操作。

- **CMO_TYPE_PREFETCH**：预取操作，将数据从全局内存提前加载到L2缓存
- **CMO_TYPE_WRITEBACK**：写回操作，将L2缓存中的修改数据写回全局内存，同时在缓存中保留副本
- **CMO_TYPE_INVALID**：失效操作，丢弃L2缓存中的数据块
- **CMO_TYPE_FLUSH**：刷新操作，强制将L2缓存数据写回全局内存并从缓存中移除

#### SDMA Quiet接口（SHMEM扩展接口）

```c
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(AscendC::LocalTensor<T> &buf, uint32_t sync_id);
```

- **功能**：等待STARS队列中的操作任务完成，用于同步
- **参数说明**：
  - `buf`：临时UB缓冲区地址
  - `ub_size`：UB缓冲区大小
  - `sync_id`：同步ID
- **特点**：通过下SDMA的Flag任务，并轮询Flag等待STARS队列中的操作完成

## 环境要求

### 硬件要求
- 昇腾AI处理器（Atlas 200I A2/A3、Atlas 300T A2/A3等）
- 架构兼容：aarch64、x86

### 软件依赖
参考[SHMEM软件依赖](https://gitcode.com/cann/shmem#%E8%BD%AF%E4%BB%B6%E4%BE%9D%E8%B5%96)，配置支持SDMA功能的相关CANN版本。

### 功能依赖说明

**重要**：本示例中的Device侧CMO接口`aclshmemx_cmo_nbi`依赖SDMA功能，需要参考example/sdma或example/cmo，配置`attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA`以启动SDMA引擎。

## 编译步骤

### 1. 编译并安装SHMEM软件包

```bash
cd shmem/
bash scripts/build.sh -package
./install/*/SHMEM_1.0.0_linux-*.run --install
source install/set_env.sh
```

### 2. 编译示例程序

```bash
cd shmem/
bash scripts/build.sh -examples
```

编译成功后，可执行文件位于：`build/bin/cmo`

## 运行方法

```bash
cd shmem/examples/cmo
bash run.sh -pes ${PEs} -type ${TYPE}
```

### 参数说明

- PEs：指定用于运行的设备（NPU）数量，限定单台机器内。
- TYPE：指定传输数据类型，当前支持：int，uint8，int64，fp16，fp32。

### 运行示例：使用2个NPU测试int类型数据
```bash
bash run.sh -pes 2 -type int
```

## 输出结果

### 控制台输出

程序运行时会输出每个PE的完成信息：
```
PE 0 Finished!
PE 1 Finished!
[SUCCESS] demo run success in pe 0
[SUCCESS] demo run success in pe 1
```

### CSV文件输出

程序会在`output/`目录下生成以下CSV文件：

#### 1. `{PE_ID}_band.csv` - 带宽性能测试结果

包含以下列：
- `loop_times`: 循环次数（默认100次）
- `copy_size_per_loop`: 每次循环拷贝的数据大小（小于L2 Cache大小以验证整块预取效果）
- `blocks`: 使用的block数量
- `copypad_size`: 单次DataCopy操作的数据大小
- `no_prefetch_time/us`: 无预取，平均拷贝时间（微秒）
- `no_prefetch_band/Gbps`: 无预取，平均拷贝带宽（GB/s）
- `host_prefetch_time/us`: Host侧整块预取后，平均拷贝时间（微秒）
- `host_prefetch_band/Gbps`: Host侧预取后，平均拷贝带宽（GB/s）
- `device_block_prefetch_time/us`: Device块预取后，平均拷贝时间（微秒）
- `device_block_prefetch_band/Gbps`: Device块预取后，平均拷贝带宽（GB/s）

#### 2. `{PE_ID}_cmo.csv` - CMO操作延迟测试结果

包含以下列：
- `loop_times`: 循环次数（默认100次）
- `blocks`: 使用的block数量
- `cmo_size`: CMO操作的数据大小
- `cmo_send_time_p05/us`: CMO发送时间的5%分位数（微秒）
- `cmo_send_time_p50/us`: CMO发送时间的50%分位数（微秒）
- `cmo_send_time_p95/us`: CMO发送时间的95%分位数（微秒）
- `cmo_flag_time_p05/us`: CMO同步等待时间的5%分位数（微秒）
- `cmo_flag_time_p50/us`: CMO同步等待时间的50%分位数（微秒）
- `cmo_flag_time_p95/us`: CMO同步等待时间的95%分位数（微秒）

### 性能指标说明

- **带宽**：衡量数据传输速率，单位为GB/s
- **延迟**：衡量操作完成时间，单位为微秒
- **分位数**：用于统计分布情况，p50为中位数。

## 参考文档

- [CANN应用开发接口文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/appdevg/acldevg/acldevg_0001.html)
- [内存管理 aclrtCmoAsync](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/appdevgapi/aclcppdevg_03_0123.html)