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
./build/bin/udma_atomic_add 2 0 tcp://127.0.0.1:8899 2 0 0 & # PE 0
./build/bin/udma_atomic_add 2 1 tcp://127.0.0.1:8899 2 0 0 & # PE 1
```

3.命令行参数说明
```bash
./udma_atomic_add <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>
```

- n_pes: 全局PE数量。
- pe_id: 当前进程的PE号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>
- g_npus: 当前机器上启动的NPU卡的数量。
- f_pe: 当前机器上使用的第一个PE号。
- f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号
