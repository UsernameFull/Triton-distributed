# SHMEM Env Vars

## 初始化相关
使用unique id的接口初始化时，需要手动配置环境变量SHMEM_UID_SESSION_ID或者SHMEM_UID_SOCK_IFNAME，同时配置时只读SHMEM_UID_SESSION_ID。指定SHMEM_UID_SESSION_ID时需保证ip可连通，port空闲。指定SHMEM_UID_SOCK_IFNAME时需保证网口指定的网络协议地址存在。

* `SHMEM_UID_SESSION_ID`:直接指定PE 0的监听socket的ip和端口。
SHMEM_UID_SESSION_ID配置示例：
SHMEM_UID_SESSION_ID=127.0.0.1:1234
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886

* `SHMEM_UID_SOCK_IFNAME`:指定PE 0的监听socket的网口名和网络层协议。
SHMEM_UID_SOCK_IFNAME配置示例：
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  取ipv4
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  取ipv6

## 多实例相关
由于每个实例都有独立的bootstrap，每个bootstrap构建时需要提供一个可用端口

* `SHMEM_INSTANCE_PORT_RANGE`:直接指定可用的端口范围。
SHMEM_INSTANCE_PORT_RANGE配置示例：
export SHMEM_INSTANCE_PORT_RANGE=1024:2047

## 日志相关

日志相关环境变量及详细介绍见[SHMEM日志](../debug/log_debug.md)。

## Profiling相关
SHMEM提供Profiling打点工具，通过采集系统时钟周期数并转换为实际时间，精准量化不同Block（计算核）、不同Frame（埋点 ID）下的MTE搬运性能，详细介绍请参考[在示例中使用Profiling工具](../debug/profiling.md).
* `SHMEM_CYCLE_PROF_PE`: 用于设置需要进行Profiling采集的pe，pe_id设置范围[0，PEs-1]，需要取消采集请`unset SHMEM_CYCLE_PROF_PE`。