# Triton-distributed 双机测试

`test_ascend_2nodes.py` 在两台 A2/910B 上各启动一个进程、一张 NPU，
从 Triton JIT 内核调用 `triton_dist.language.extra.libshmem_device.putmem/getmem`。
传输引擎使用 ROCE，HCCL 仅用于控制同步和汇总校验状态，不承载被测数据。
测试每个元素的值，并在两个 rank 都校验通过后输出 PASS。
这是通信原语正确性测试，不是已有 MTE AllGather GEMM 示例的双机版本或性能基准。

两端需有相同的项目代码、匹配的已安装 Ascend 后端，以及启用 RDMA 的 SHMEM。
当前环境仅做了本地逻辑检查，双机 NPU 编译和执行结果需由目标机器验证。
HCCL 通信成功并不保证已安装 SHMEM 启用了 RDMA。

如果 SHMEM 尚未启用 RDMA，两台机器都在各自安装环境中构建并安装一次：

```bash
cd /workspace/Triton-distributed
source .ascend-910b/env.sh
(
  set -e
  cd 3rdparty/shmem
  COMPILE_OPTIONS=-DACLSHMEM_RDMA_SUPPORT=ON \
    bash scripts/build.sh -enable_rdma -python_extension
  python -m pip install --force-reinstall --no-deps dist/shmem-*.whl
)
```

两台机器都加载环境，并将 MASTER_IP 改为第一台机器能被第二台访问的 IPv4 地址：

```bash
cd /workspace/Triton-distributed
source .ascend-910b/env.sh
export MASTER_IP=10.0.0.1
```

第一台启动：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 timeout --kill-after=15s 600s \
  python -m torch.distributed.run \
  --nnodes=2 --node-rank=0 --nproc-per-node=1 \
  --rdzv-backend=static --master-addr="$MASTER_IP" --master-port=29501 \
  scripts/test_ascend_2nodes.py --shmem-port=8765
```

第二台同时启动：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 timeout --kill-after=15s 600s \
  python -m torch.distributed.run \
  --nnodes=2 --node-rank=1 --nproc-per-node=1 \
  --rdzv-backend=static --master-addr="$MASTER_IP" --master-port=29501 \
  scripts/test_ascend_2nodes.py --shmem-port=8765
```

选用实际空闲的物理卡替换 `ASCEND_RT_VISIBLE_DEVICES=0`，两个进程内部均使用本地逻辑卡 0。
29501 为 torchrun 端口，8765 为独立 SHMEM bootstrap 端口，需互通且空闲；
设备侧还需已配置的 RoCE 链路。两台都输出 `PASS: Triton-distributed two-node put/get`
才表示成功，默认执行三轮 put/get。超时或任何数值不匹配均不算通过。
