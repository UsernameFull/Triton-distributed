### 使用方式

#### 1. 编译项目

**default以外的流程执行需要自行安装并导入MPI环境变量**[安装参考](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/hccltool/HCCLpertest_16_0002.html)
该用例基于mpich实现，安装Open MPI或其他MPI如果遇到报错可能需要自行调整脚本中mpi相关参数。

```bash
# mpich安装在默认路径时可参考，需根据实际情况替换。
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```

在 `shmem/` 根目录下执行编译脚本：

```bash
bash scripts/build.sh
source install/set_env.sh
```

##### 2. 运行 init 用例

用例目录
```bash
cd examples/init
```

用例接收两个参数，第一个参数指定流程，第二个参数指定pe数量（不能超过卡数）。不指定时默认`mode = default`,`pesize = 2`。
执行 default 流程，2 pe
```bash
bash run.sh -mode default -pesize 2
```

执行 mpi 流程，2 pe
```bash
bash run.sh -mode mpi -pesize 2
```

执行 uid 流程，2 pe
```bash
bash run.sh -mode uid -pesize 2
```

执行 uid_multi 流程，4 pe
```bash
# enpxxx需要替换为ip addr指令获得的网卡
export SHMEM_UID_SOCK_IFNAME=enpxxxxxxx:inet4

bash run.sh -mode uid_multi -pesize 4

unset SHMEM_UID_SOCK_IFNAME
```

执行 uid_default 流程，2 pe
```bash
bash run.sh -mode uid_default -pesize 2
```

##### 3. 跨机运行
执行 default 流程，双机，每个机器2 pe，在两台机器分别执行如下命令。
```bash
# pe0所在机器
bash run.sh -mode default -pesize 4 -fpe 0 -gnpus 2 -ipport ${该机器的ip:port}
# 其他机器
bash run.sh -mode default -pesize 4 -fpe 2 -gnpus 2 -ipport ${pe0机器的ip:port}
```

执行 mpi/uid 流程，双机，每个机器2 pe。
这两个流程依赖mpi能力，跨机需自行配置hostfile文件，以mpich配置文件为例
```sh
# 请替换成实际机器的ip，且保证pe0机器在最前面,冒号后配置每个机器支持启动的pe数，pe数需小于卡数，建议每台机器启动pe数量一致。
0.0.0.1:2
0.0.0.2:2
```
**mpirun要求多机可执行文件位置一致，请先用相同mode编译所有机器上的样例。如果mode不一致会导致报错或者卡死。**
```sh
bash run.sh -build -mode mpi #uid流程则讲mode参数改为uid
```
然后在pe0执行
mpi流程
```bash
# gnpu参数用来指定每台设备自身的pe数，在所有机器启动卡数一致时，设置为单机pe数量可以保证都从0卡开始启动。
bash run.sh -mode mpi -gnpus 2
```
uid流程
```bash
bash run.sh -mode uid -gnpus 2 -ipport ${pe0机器的ip:port}
```
**如果想再次以 mpi/uid 模式执行单机用例推荐删除hostfile文件**