使用方式:

1. 在shmem/目录编译:
```
bash scripts/build.sh -examples
```
2. 在shmem/examples/allgather目录执行demo:
```
# 完成PEs卡下的allgather同时验证精度，性能数据会输出在result.csv中。
# PEs : [2, 4, 8]
# TYPEs : [int, int32_t, float16_t, bfloat16_t]
bash run.sh -pes ${PEs} -type ${TYPEs}
```