# Root Info Generate 工具 - 用户指南

## 概述

**root_info_generate** 是一个辅助工具，用于查看Ascend950 NPU设备拓扑地址信息，帮助用户了解组网结构和设备配置。

**重要说明**：
- **支持范围**：当前仅支持Ascend950平台
- **自动生成**：root info信息会在SHMEM初始化时自动调用相关接口生成，无需用户手动执行工具
- **辅助用途**：本工具主要用于辅助用户查看和了解组网拓扑、EID地址等信息，便于开发和调试
- **非必需操作**：正常使用SHMEM时，用户不需要手动运行此工具

## 背景知识

### 简介

昇腾芯片在950代际中，超平面使用Unified Bus（UB）总线组网，在不同产品形态中使用多种不同的拓扑组网方式。本模块用于发现不同拓扑下每条边的端点地址。

#### 组网介绍

组网主要采用了MESH和CLOS两种类型组网。相关介绍可参考论文：https://arxiv.org/abs/2503.20377

##### MESH组网

每个NPU之间均有一条直连的物理链路，因此有一对独立的通信地址。

**示例**：在同一个NPU板上有8个NPU，因此存在 `8*7/2 = 28` 条物理路径，即28对通信地址。

**特点**：
- 直连链路，无交换芯片
- 通信质量最高，时延最低
- 地址数量：N*(N-1)/2 对地址（N为NPU数量）

##### CLOS组网

任意两个NPU之间通过交换芯片转发，因此一个NPU只需一个地址。

**多平面设计**：在常见组网中，由于可靠性等原因，CLOS组网通常分为多个平面，每个平面对应一个地址。

**示例**：在液冷POD中，NPU使用两个独立的逻辑口连接两个分开的网络平面。

**特点**：
- 通过交换芯片转发
- 地址数量与平面数量相同
- 可实现流量分担和冗余

##### 组网规划

#### 网络层级说明

在950芯片代际中，按通信质量和范围，将网络划分为多层：

| 网络层级 | 说明 |
|:-------|:-----------|
| **层级0** | 通信质量最高，时延最低。多为MESH组网，主要是同一个NPU单板内的fullmesh网络和POD形态的框级网络 |
| **层级1** | 通信质量次高，时延中等。为CLOS组网，通信范围较大，为超节点范围，仍然在scale up范围内 |
| **层级2** | 通信质量最低，CLOS组网，通信范围为整个集群，主要是ROCE或者UBOE这类scale out网络 |

**层级特点对比**：

- **层级0**：板内/框内通信，最优质量
- **层级1**：超节点内通信，中等质量  
- **层级2**：集群间通信，最低质量但范围最大

#### 网络地址说明

| 网络层级 | 地址规划说明 |
|:-------|:-----------|
| **层级0** | 由于是MESH组网为主，因此有多对通信地址。在拓扑信息中表示为每个NPU上每个端口的地址。**地址类型：EID** |
| **层级1** | 根据组网平面填写地址，多平面组网时，地址数量与平面数量相同。集合通信在不同平面之间做流量分担。**地址类型：EID** |
| **层级2** | 地址规划与层级1相同。**地址类型：IP地址** |

**地址类型说明**：

- **EID (Extended ID)**：扩展标识符，用于层级0和层级1的UB网络通信
- **IP地址**：用于层级2的scale out网络通信（如RoCE、UBOE）

#### EID地址格式

EID地址为16字节（128位），在JSON输出中表示为32位十六进制字符串：

```
格式示例: "000000000000006000100000dfdf008b"
长度: 32个十六进制字符 = 16字节
```

**EID结构**（参考）：
- 包含subnet_prefix（子网前缀）
- 包含interface_id（接口标识）
- 用于UB网络的端点寻址

## 功能特性

- **拓扑查询**: 获取指定NPU的完整拓扑地址信息
- **缓冲区检测**: 自动获取拓扑数据所需缓冲区大小
- **文件路径查询**: 获取拓扑配置文件路径
- **JSON格式输出**: 以结构化JSON格式显示拓扑信息

## 前置条件

### 系统要求
- **SOC类型**: Ascend950
- **平台**: Linux
- **依赖库**: libdcmi.so驱动库、c_sec安全库

### 编译要求
- CMake 3.16+
- C++17编译器
- Ascend驱动已安装

## 安装说明

### 从源码编译

```bash
# 克隆代码仓库
git clone <仓库地址>
cd shmem

# 使用Ascend950 SOC类型编译
bash scripts/build.sh -soc_type Ascend950

# 工具生成位置：
# - 编译后（立即可用）: build/bin/root_info_generate
# - 安装后（完整安装）: install/shmem/bin/root_info_generate
```

### 验证安装

编译完成后可立即使用：
```bash
# 检查编译生成的工具
ls -l build/bin/root_info_generate

# 直接运行（编译后）
./build/bin/root_info_generate <物理ID>

# 或使用安装后的工具（完整编译安装后）
./install/shmem/bin/root_info_generate <物理ID>
```

### 打包安装

完整编译安装后生成安装包：
```
install/aarch64/SHMEM_1.0.0_linux-aarch64.run
```

## 使用方法

### 基本语法

```bash
./root_info_generate <物理ID>
```

**参数说明**:
- `物理ID`: NPU物理ID（整数，范围：0-63）

### 使用示例

#### 示例1：查询ID为3的NPU

```bash
./root_info_generate 3
```

**预期输出**:
```
Generating root info for NPU with physical ID: 3
Required buffer size: 2048 bytes
topo_addr_info_get succeeded, actual size: 1329 bytes
Rank info:
{"version": "2.0","topo_file_path": "/usr/local/Ascend/driver/topo/950/atlas_850_1.json","rank_count": 1,"rank_list": [{"device_id": 3,"local_id": 3,"level_list": [{"net_layer": 0,"net_instance_id": "sp-1_srv65535","net_type": "MESH","net_attr": "","rank_addr_list": [{"addr_type": "EID","addr": "000000000000006000100000dfdf008b","plane_id": "plane_1","ports": ["1/0"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf00cb","plane_id": "plane_1","ports": ["1/8"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf00c3","plane_id": "plane_1","ports": ["1/7"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf00ab","plane_id": "plane_1","ports": ["1/4"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf00a3","plane_id": "plane_1","ports": ["1/3"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf009b","plane_id": "plane_1","ports": ["1/2"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf0093","plane_id": "plane_1","ports": ["1/1"]}]}, {"net_layer": 1,"net_instance_id": "superpod_-1","net_type": "CLOS","net_attr": "","rank_addr_list": [{"addr_type": "EID","addr": "000000000000004000100000dfdf00df","plane_id": "plane_1","ports": ["1/5","1/6"]},{"addr_type": "EID","addr": "000000000000006000100000dfdf005f","plane_id": "plane_0","ports": ["0/4","0/5","0/6","0/7"]}]}]}]}
Topology file path: /usr/local/Ascend/driver/topo/950/atlas_850_1.json
Root info generation completed successfully
```

#### 示例2：查询其他NPU

```bash
./root_info_generate 0
./root_info_generate 5
```

#### 示例3：无效物理ID

```bash
./root_info_generate 100
```

**错误输出**:
```
Generating root info for NPU with physical ID: 100
Error: topo_addr_info_get_size failed with ret=-1
```

## 输出解读

### 成功输出结构

工具执行成功后会输出以下信息：

1. **缓冲区大小**
   ```
   Required buffer size: XXX bytes
   ```

2. **实际数据大小**
   ```
   topo_addr_info_get succeeded, actual size: XXX bytes
   ```

3. **JSON拓扑数据**（紧凑格式，建议使用jq格式化查看）
   - 输出为JSON字符串，包含拓扑地址信息
   - 可使用 `jq '.'` 格式化输出便于阅读

4. **拓扑文件路径**
   ```
   Topology file path: /usr/local/Ascend/driver/topo/950/atlas_850_1.json
   ```

### 错误处理

当工具执行失败时会输出错误信息：
```
Error: topo_addr_info_get_size failed with ret=-1
```

常见错误原因：
- 物理ID超出范围（有效范围0-63）
- 驱动未初始化或设备不可访问
- 设备权限不足

## 常见问题与解决方案

### 问题1：工具找不到

**症状**:
```bash
./root_info_generate: command not found
```

**解决方案**:
1. 使用SOC_TYPE=Ascend950编译
2. 验证安装位置: `build/bin/` 或 `install/shmem/bin/`
3. 检查编译日志确认工具已生成

### 问题2：库依赖缺失

**症状**:
```bash
error while loading shared libraries: libshmem_rootinfo.so: cannot open shared object file
```

**解决方案**:
```bash
# 设置库路径
export LD_LIBRARY_PATH=install/shmem/lib:$LD_LIBRARY_PATH

# 或复制库到标准位置
cp install/shmem/lib/*.so /usr/local/lib/
ldconfig
```

### 问题3：驱动未初始化

**症状**:
```bash
Error: topo_addr_info_get_size failed with ret=-1
```

**解决方案**:
1. 检查Ascend驱动安装: `ls /usr/local/Ascend/`
2. 初始化驱动: 运行驱动初始化脚本
3. 验证NPU设备: `ls /dev/davinci*`

### 问题4：权限拒绝

**症状**:
```bash
Error: cannot access /dev/davinci0
```

**解决方案**:
```bash
# 检查设备权限
ls -l /dev/davinci*

# 将用户添加到相应组
sudo usermod -a -G ascend <用户名>

# 或使用适当权限运行
sudo ./root_info_generate 0
```

### 问题5：无效物理ID

**症状**:
```bash
Error: get_mainboard_id returned null for phy_id=100
```

**解决方案**:
- 使用有效的物理ID范围: 0到ACLSHMEMI_MAX_NPU_COUNT-1
- 检查可用NPU: `ls /dev/davinci*`

## 支持与故障排查

### 获取帮助

1. **检查日志**: 仔细审查错误消息
2. **验证安装**: 确保所有依赖已安装
3. **测试驱动**: 验证Ascend驱动功能
4. **联系支持**: 提交包含详细日志的问题

### 调试模式

启用详细日志用于调试：
```bash
# 设置调试环境（如果可用）
export SHMEM_LOG_LEVEL=DEBUG
./root_info_generate 0
```

### 日志分析

检查系统日志获取额外信息：
```bash
# 检查驱动日志
dmesg | grep -i ascend

# 检查应用日志
journalctl -u ascend-driver
```

