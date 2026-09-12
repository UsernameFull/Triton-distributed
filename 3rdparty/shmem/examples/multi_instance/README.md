本example主要为了模拟多实例下的实例创建和算子执行流程，会进行数次不同实例的创建以及释放，同时每个实例都会运行一次allgather算子，并验证精度。

注意：本example执行依赖多个端口，默认执行脚本会将预留端口设置为1024-2047，可在run.sh中的SHMEM_INSTANCE_PORT_RANGE环境变量中自定义。

使用方式:

1. 在shmem/目录编译:
```
bash scripts/build.sh -examples
```
2. 在shmem/examples/multi_instance目录执行demo:
```
# 完成PEs卡下的allgather同时验证精度。
# PEs : [4, 8]
# TYPEs : [int, int32_t, float16_t, bfloat16_t]
bash run.sh -pes ${PEs} -type ${TYPEs}
```