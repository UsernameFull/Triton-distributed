# 快速开始

## 介绍
本系统主要面向昇腾平台上的模型和算子开发者，提供便携易用的多机多卡内存访问方式，方便用户开发在卡间同步数据，加速通信或通算融合类算子开发。  

## 软件架构
共享内存库接口主要分为host和device接口部分：
- host侧接口提供初始化、内存管理、通信域管理以及同步功能。
- device侧接口提供内存访问、同步以及通信域管理功能。

## 目录结构说明
详细介绍见[code_organization](code_organization.md)
``` 
├── 3rdparty // 依赖的第三方库
├── docs     // 文档
├── examples // 使用样例
├── include  // 头文件
├── scripts  // 相关脚本
├── src      // 源代码
└── tests    // 测试用例
```

## 软件硬件配套说明
- 硬件型号支持 
  - Atlas 800I A2/A3 系列产品
  - Atlas 800T A2/A3 系列产品
- 平台：aarch64/x86
- 配套软件：驱动固件 Ascend HDK 25.0.RC1.1、 CANN 8.3.RC1 及之后版本。（参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)》安装CANN开发套件包以及配套固件和驱动）  
cmake >= 3.19  
GLIBC >= 2.28

## 快速上手
 - 设置CANN环境变量<br>
    ```sh
    # root用户安装（默认路径）
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
 - 共享内存库编译<br>
    编译共享内存库，设置共享内存库环境变量：
    ```sh
    cd shmem
    bash scripts/build.sh
    source install/set_env.sh
    ```
 - run包使用<br>
    软件包名为：SHMEM_{version}_linux-{arch}.run <br>
    其中，{version}表示软件版本号，{arch}表示CPU架构。<br>
    安装run包（需要依赖cann环境）<br>

    ```sh
    chmod +x 软件包名.run # 增加对软件包的可执行权限
    ./软件包名.run --check # 校验软件包安装文件的一致性和完整性
    ./软件包名.run --install # 安装软件，可使用--help查询相关安装选项
    ```
    出现提示`xxx install success!`则安装成功

shmem 默认开启tls通信加密。如果需要关闭，需要调用接口主动关闭：
```c
int32_t ret = aclshmemx_set_conf_store_tls(false, null, 0);
```
具体细节详见安全声明章节

执行一个样例matmul_allreduce算子。  
1.在shmem/目录编译:

```sh
bash scripts/build.sh -examples
```

2.在shmem/examples/matmul_allreduce目录执行demo:

```sh
bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192
```
注：example及其他样例代码仅供参考，在生产环境中请谨慎使用。
```
## 功能自测用例

 - 共享内存库接口单元测试
在工程目录下执行
```sh
bash scripts/build.sh -uttests
bash scripts/run.sh
```
run.sh脚本提供-ranks -ipport -test_filter等参数自定义执行用例的卡数、ip端口、gtest_filter等  

例

```sh
# 8卡，ip:port 127.0.0.1:8666，运行所有*Init*用例
bash scripts/run.sh -ranks 8 -ipport tcp://127.0.0.1:8666 -test_filter Init
```

## python侧test用例     [python接口API列表](api/pythonAPI.md)
1. 在scripts目录下编译的时候，带上build python的选项

```sh
bash build.sh -python_extension
```

2. 在install目录下，source环境变量

```sh
source set_env.sh
```

3. 在src/python目录下，进行setup，获取到wheel安装包

```sh
python3 setup.py bdist_wheel
```

4. 在src/python/dist目录下，安装wheel包

```sh
pip3 install shmem-xxx.whl --force-reinstall
```

5. 设置是否开启TLS认证，默认开启，若关闭TLS认证，请使用如下接口

```python
import shmem as shm
shm.set_conf_store_tls(False, "")   # 关闭tls认证
```

```python
import shmem as shm
tls_info = "xxx"
shm.set_conf_store_tls(True, tls_info)      # 开启TLS认证
```

6. 使用torchrun运行测试demo

```sh
torchrun --nproc-per-node=k test.py // k为想运行的ranksize
```
看到日志中打印出“test.py running success!”即为demo运行成功

## unique id 初始化方式

注：使用unique id的接口初始化，需要手动配置环境变量SHMEM_UID_SESSION_ID或者SHMEM_UID_SOCK_IFNAME，同时配置时只读SHMEM_UID_SESSION_ID
SHMEM_UID_SESSION_ID配置示例：
SHMEM_UID_SESSION_ID=127.0.0.1:1234
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886
SHMEM_UID_SOCK_IFNAME配置示例：
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  取ipv4
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  取ipv6
不配置默认取eth:inet4

- python初始化例子
```python
import shmem as ash

# xxx

uid = ash.aclshmem_get_unique_id()
ret = ash.aclshmem_init_using_unique_id(rank, world_size, mem_size, uid)

# xxx
```

准备以上启动代码`init.py`后，使用`torchrun --nproc-per-node 8 init.py`，其中进程数 8 根据实际需要改动，更详细的例子，请参考`unique_id_test.py`文件。

- c++初始化例子
```cpp
aclshmemx_uniqueid_t uid;
aclshmemx_init_attr_t *attr;
int ret = aclshmemx_get_uniqueid(&uid);
ret = aclshmemx_set_attr_uniqueid_args(my_pe, n_pes, mem_size, &uid, attr);
```
## SHMEM方式
注：使用unique id的接口初始化，可以手动配置环境变量SHMEM_UID_SESSION_ID或者SHMEM_UID_SOCK_IFNAME，同时配置时只读SHMEM_UID_SESSION_ID，都不配置会自动搜索可用网口。
SHMEM_UID_SESSION_ID配置示例：
SHMEM_UID_SESSION_ID=127.0.0.1:1234
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666%eth]:886
SHMEM_UID_SOCK_IFNAME配置示例：
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  取ipv4
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  取ipv6
不配置默认取inet4自动搜索可用网口，搜索优先级：非docker、lo>>docker>>lo。


- c++初始化例子
```cpp
aclshmemx_uniqueid_t uid;
aclshmemx_init_attr_t *attr;
int ret = aclshmemx_get_uniqueid(&uid);
shmemx_set_attr_uniqueid_args(rank, rank_size, local_mem_size, &uid, &attributes);
status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, attributes);
```