## 环境要求
同[rdma_demo](../rdma_demo/README.md)中的环境要求。

## 使用方式
1.在shmem/目录编译:
```bash
bash scripts/build.sh -enable_rdma -examples
```
2.直接在`examples/rdma_perftest`目录下执行`bash run.sh`；或者在shmem/目录运行:
- 单机2卡执行命令
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_perftest 2 0 tcp://127.0.0.1:8899 2 0 0 highlevel_put_pingpong_latency 64 # PE 0
    ./build/bin/rdma_perftest 2 1 tcp://127.0.0.1:8899 2 0 0 highlevel_put_pingpong_latency 64 # PE 1
    ```
    > 注：\<shmem-root-directory\>为SHMEM项目的根目录。
- 跨机2卡执行命令

    假设机器A的ip为ip1，机器B的ip为ip2。
    在机器A执行如下命令：
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_perftest 2 0 tcp://ip1:8765 1 0 0 highlevel_put_pingpong_latency 64 # PE 0
    ```
    同时，在机器B执行如下命令：
    ```bash
    export PROJECT_ROOT=<shmem-root-directory>
    export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
    ./build/bin/rdma_perftest 2 1 tcp://ip1:8765 1 1 0 highlevel_put_pingpong_latency 64 # PE 1
    ```
    > 注：\<shmem-root-directory\>为SHMEM项目的根目录。

3.命令行参数说明
```bash
    ./rdma_perftest <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> <test_type> <msg_len>
```
- n_pes: 全局PE数量，只支持2个PE。
- pe_id: 当前进程的PE号。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。如果执行跨机测试，需要将IP设为PE0所在Host的IP。
- g_npus: 当前机器上启动的NPU卡的数量。
- f_pe: 当前机器上使用的第一个PE号。
- f_npu: 当前机器执行本样例使用的第一张NPU卡的卡号
- test_type: 测试类型。
    - highlevel_put_pingpong_latency：测试Put高阶接口的pingpong时延。
    - postsend_cost: 测试postsend接口耗时。
    - highlevel_put_bw: 测试Put高阶接口的带宽。
    - rdma_mte_bw: 测试并行下发MTE和RDMA时的带宽。
- msg_len: 测试传输的数据量大小，单位为字节（Byte）。