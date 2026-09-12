# shmem_perftest

## 示例概述

shmem_perftest是一个用于**测试shmem MTE（Memory Transfer Engine）性能**的参数化测试示例，支持多种测试类型、多种数据类型和灵活的参数配置。该示例通过[SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h)宏采集性能数据，测试不同核数和数据量下的MTE传输带宽，帮助用户评估MTE数据传输性能。**该脚本测试结果仅做参考，性能以实际场景为准**

## 测试目的

本示例主要用于测试以下MTE数据传输操作的性能：
1. **单向Put**: 仅PE0执行put操作，将数据传输到远端PE
2. **双向Put**: 两个PE同时执行put操作，互相传输数据
3. **单向Get**: 仅PE0执行get操作，从远端PE读取数据
4. **双向Get**: 两个PE同时执行get操作，互相读取数据

通过测试结果，用户可以了解MTE接口在不同场景下的性能表现。

## 功能特性

### 支持的测试类型

- **put**: 单向put操作，仅环境变量`SHMEM_CYCLE_PROF_PE`指定的PE执行数据搬运，未指定时默认PE0
- **bi_put**: 双向put操作，两个PE都执行数据搬运
- **get**: 单向get操作，仅环境变量`SHMEM_CYCLE_PROF_PE`指定的PE执行数据搬运，未指定时默认PE0
- **bi_get**: 双向get操作，两个PE都执行数据搬运
- **all**: 运行所有测试类型

### 支持的数据类型

- **浮点型**: `float`
- **有符号整型**: `int8`, `int16`, `int32`, `int64`
- **无符号整型**: `uint8`, `uint16`, `uint32`, `uint64`
- **字符型**: `char`
- **all**: 测试所有数据类型

### 其他特性

- ✅ 可配置核数范围（默认32核）
- ✅ 可配置数据量范围（支持2的幂次方）
- ✅ 可设置循环次数以获得稳定性能数据
- ✅ 使用SHMEMI_PROF采集各核性能数据
- ✅ 生成CSV格式性能报告
- ✅ 支持批量测试（"all"选项）

## 编译说明

在shmem根目录下编译示例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples

# 如需运行 --memory-type dram，需启用CANN模式编译
bash scripts/build.sh -examples -cann
```

## 使用方法

### 基本用法

```bash
# 进入示例目录
cd examples/mte_perftest/shmem_perftest/

# 运行测试脚本
./run.sh [选项]
```

### 命令行参数

| 参数 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--test-type <type>` | `-t <type>` | 测试类型 (put/bi_put/get/bi_get/all) | put |
| `--datatype <type>` | `-d <type>` | 数据类型 (float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char/all) | float |
| `--block-size <size>` | `-b <size>` | 设置单个核数 | 32 |
| `--block-range <min> <max>` | - | 设置核数范围 | 32-32 |
| `--exponent <exponent>` | `-e <exponent>` | 设置数据量的幂数 (2^exponent) | - |
| `--exponent-range <min> <max>` | - | 设置数据量的幂数范围 | 3-17 |
| `--loop-count <count>` | - | 设置循环次数 | 1000 |
| `--memory-type <hbm\|dram>` | - | 设置SHMEM内存类型，dram使用HOST_SIDE内存 | hbm |
| `-pes <size>` | - | 设置PE数量，暂不支持修改 | 2 |
| `-ipport <ip:port>` | - | 设置通信地址 | tcp://127.0.0.1:8764 |
| `-gnpus <num>` | - | 设置NPU数量，暂不支持修改 | 2 |
| `-fnpu <id>` | - | 设置首个NPU ID，控制起始NPU | 0 |
| `-fpe <id>` | - | 设置首个PE ID，暂不支持修改 | 0 |

### DRAM内存测试约束

`--memory-type dram` 会使用 `aclshmemx_malloc(..., HOST_SIDE)` 分配Host侧DRAM内存。该功能依赖CANN模式，编译时必须使用：

```bash
bash scripts/build.sh -examples -cann
```

DRAM测试需要运行环境支持Host侧DRAM内存访问，相关硬件和可用内存约束可参考 [rma_d2h_demo](../../rma_d2h_demo/README.md) 的“约束限制”章节。shmem_perftest默认配置1GB本地内存；当测试参数需要更大本地内存时，程序会按数据量自动上调，运行前需确保可用DRAM空间大于实际本地内存配置。

### 使用示例

#### 1. 测试单个操作类型和数据类型

```bash
# 测试单向PUT操作，float类型，32核，数据量从2^8到2^20字节
./run.sh -t put -d float --block-size 32 --exponent-range 8 20 --loop-count 1000

# 测试双向GET操作，int32类型
./run.sh -t bi_get -d int32 --block-size 32 --exponent-range 8 20 --loop-count 1000
```

#### 2. 测试所有操作类型（指定数据类型）

```bash
# 测试所有操作类型（put/bi_put/get/bi_get），使用float数据类型
./run.sh -t all -d float --block-size 32 --exponent-range 8 20 --loop-count 1000
```

#### 3. 测试所有数据类型（指定操作类型）

```bash
# 测试PUT操作，使用所有数据类型
./run.sh -t put -d all --block-size 32 --exponent-range 8 20 --loop-count 1000
```

#### 4. 测试所有操作类型和数据类型

```bash
# 测试所有操作类型和所有数据类型
./run.sh -t all -d all --block-size 32 --exponent-range 8 20 --loop-count 1000
```

#### 5. 使用默认参数运行

```bash
# 使用默认参数运行put测试
./run.sh

# 运行所有测试类型
./run.sh -t all
```

#### 6. 图形化数据

```bash
# 使用perf_data_process.py -d 传入csv数据存储路径，图形化结果将存储在csv存储路径下picture文件夹下。
python ../utils/perf_data_process.py -d output
```
## 输出说明

### CSV文件格式

测试运行后，会在当前目录下生成`output`文件夹，其中包含测试结果的CSV文件，文件名格式为：

```
{test_type}_{data_type}_{prof_pe}.csv
```

例如：
- `put_float_0.csv` - 单向PUT操作，float类型
- `bi_get_int32_0.csv` - 双向GET操作，int32类型

### CSV文件内容

CSV文件包含以下字段：

| 字段 | 描述 |
|------|------|
| `DataSize/B` | 数据大小（字节） |
| `Npus` | 使用的NPU数量 |
| `Blocks` | 核数 |
| `Bandwidth/GBps` | 带宽大小（GB/s） |
| `CoreMaxTime/us` | 最大核时间（微秒） |
| `SingleCoreTime/us` | 单核时间（微秒），每个核一列 |

## 性能采集说明

本示例使用`SHMEMI_PROF_START/END`宏进行性能数据采集：

- **采集PE**: 通过环境变量`SHMEM_CYCLE_PROF_PE`指定，默认为0
- **最大记录核数**: `ACLSHMEM_CYCLE_PROF_MAX_BLOCK`（32核）
- **最大记录帧数**: `ACLSHMEM_CYCLE_PROF_FRAME_CNT`（32帧）

只有环境变量指定的PE会输出CSV文件，其他PE的数据不会被采集。

## 注意事项

1. **环境配置**: 运行前请确保已正确设置Ascend环境变量：
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. **数据量计算**: 测试数据量的计算方式为：`2^exponent` 字节

3. **参数范围**: 
   - 核数范围超过`ACLSHMEM_CYCLE_PROF_MAX_BLOCK`（32核）不做记录
   - 帧数不能超过`ACLSHMEM_CYCLE_PROF_FRAME_CNT`（32帧）

4. **内存要求**: 大数据量测试需要足够的设备内存

5. **性能稳定性**: 建议使用较大的循环次数（如1000次）以获得稳定的性能数据

6. **单向/双向区别**:
   - 单向操作（put/get）：只有`SHMEM_CYCLE_PROF_PE`指定的PE执行搬运
   - 双向操作（bi_put/bi_get）：两个PE都执行搬运


### 常见问题

1. **内存分配失败**
   ```
   错误: aclrtMalloc failed
   ```
   解决方案：减小数据量范围或核数，确保设备有足够内存

2. **CSV文件无数据**
   解决方案：检查`SHMEM_CYCLE_PROF_PE`环境变量是否正确设置，确保指定的PE有执行测试

3. **frame_id超过最大值**
   ```
   警告: frame_id 超过最大值 32, 停止测试
   ```
   解决方案：减小测试范围，确保测试帧数不超过32。
   
   测试帧数计算公式：
   ```
   测试帧数 = (max_block_size - min_block_size + 1) × (max_exponent - min_exponent + 1)
   ```
   
   例如：`--block-range 16 32 --exponent-range 10 20` 的测试帧数为：
   ```
   (32 - 16 + 1) × (20 - 10 + 1) = 17 × 11 = 187 > 32（超出限制）
