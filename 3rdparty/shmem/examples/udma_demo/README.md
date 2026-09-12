使用方式:

1.在shmem/目录编译:
```bash
bash scripts/build.sh -examples -soc_type Ascend950
```

2.在shmem/目录运行:
```bash
export PROJECT_ROOT=<shmem-root-directory>
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
export SHMEM_UID_SESSION_ID=127.0.0.1:8899

bash examples/udma_demo/run.sh 0 # allgather测试
bash examples/udma_demo/run.sh 1 # put signal 测试
```
默认按单机8卡启动，脚本依次拉起`PE 0`到`PE 7`，并等待所有进程退出。

3.脚本命令行参数说明
```bash
./udma_demo <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> [test_type]
```

- n_pes: 全局PE数量。
- pe_id: 当前进程对应的PE号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。
- g_npus: 当前机器上启动的NPU卡的数量。
- f_pe: 当前机器上使用的第一个PE号。
- f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号。
- test_type: 测试类型（可选），0表示运行all-gather测试（默认），1表示运行put signal测试。
