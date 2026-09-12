## 示例场景
aclGraph图结构如下：
![image.png](https://raw.gitcode.com/user-images/assets/8546182/091ff732-56c0-431e-be32-21ab6f725de4/image.png 'image.png')
由于rdma allgather只发送Write，因此在入图时需要在rdma allgather前后添加`aclshmemx_handle_wait`接口进行同步，确保所有PE上的图均已执行到相应阶段，确保aclGraph图（model）的精度正常。

## 环境要求

- 运行本示例需要机器具备RDMA环境（RDMA网卡及驱动已正确安装配置）。

### 检查RDMA环境
```bash
lspci | grep -i RDMA
for i in {0..7}; do hccn_tool -i $i -net_health -g; done
```
可用环境命令输出如下：  
![](../../docs/images/rdma_env.png)

## 使用方式
1.在shmem/目录编译:
```bash
bash scripts/build.sh -enable_rdma -examples
```
2.1 在shmem/rdma_aclgraph_demo目录运行:
```bash
bash run.sh # 单机双卡用例
```
2.2 双机8卡示例，修改对应的run.sh
```bash
# 1机
pids=()
msprof --application="./build/bin/rdma_aclgraph_demo 8 0 tcp://{1机IP}:{端口号} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 0
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 1 tcp://{1机IP}:{端口号} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 1
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 2 tcp://{1机IP}:{端口号} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 2
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 3 tcp://{1机IP}:{端口号} 4 0 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 3
pid=$!
pids+=("$pid")

# 2机
pids=()
msprof --application="./build/bin/rdma_aclgraph_demo 8 4 tcp://{1机IP}:{端口号} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 4
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 5 tcp://{1机IP}:{端口号} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 5
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 6 tcp://{1机IP}:{端口号} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 6
pid=$!
pids+=("$pid")

msprof --application="./build/bin/rdma_aclgraph_demo 8 7 tcp://{1机IP}:{端口号} 4 4 0" --output=${PROJECT_ROOT}/examples/rdma_aclgraph_demo/output/ & # pe 7
pid=$!
pids+=("$pid")
```
3.命令行参数说明
    ./rdma_aclgraph_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>

- n_pes: 全局Pe数量。
- pe_id: 当前Pe号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。如果执行跨机测试，需要将IP设为pe0所在Host的IP。
- g_npus: 当前机器上启动的NPU数量。
- f_pe: 当前机器上使用的第一个Pe号。
- f_npu: 当前机器上使用的第一个NPU卡号。