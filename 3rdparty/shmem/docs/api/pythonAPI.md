## SHMEM Python API Reference
### shmem.core API
##### 对外接口
1. 获取当前库版本。返回 ACLSHMEM 库的版本信息。

    ```python
    def get_version() -> str
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|版本信息组成的字符串，格式为 ``"libaclshmem_version=X.Y"``|

1. 生成用于 UID 初始化的唯一 ID。应由单个进程（如 rank 0）调用，并通过广播分发给其他进程。

    ```python
    def get_unique_id(empty: bool=False) -> UniqueID
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |empty|[in]|预留参数，无实际意义|
    |返回值|[out]|代表一个唯一 ID 的句柄。若生成失败则引发 ``AclshmemError``|

1. 使用唯一 ID 初始化 ACLSHMEM 运行时。这是一个集合（collective）操作，所有 PE 必须调用。

    ```python
    def init(device: int=None, uid: UniqueID=None, rank: int=None, nranks: int=None, mpi_comm=None, initializer_method: str="", mem_size: int=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |device|[in]|预留参数，无实际意义|
    |uid|[in]|用于初始化的唯一标识符，必填|
    |rank|[in]|当前进程在 ACLSHMEM 作业中的排名（0-based），必填|
    |nranks|[in]|参与 ACLSHMEM 作业的总进程数，必填|
    |mpi_comm|[in]|预留参数，无实际意义|
    |initializer_method|[in]|指定初始化方法，必须为 ``"uid"``|
    |mem_size|[in]|每个 PE 分配的对称内存大小（字节），必填|
    |返回值|-|无返回值。参数缺失引发 ``AclshmemInvalid``，初始化失败引发 ``AclshmemError``|

1. 销毁 ACLSHMEM 运行时，释放所有资源。每个进程在完成所有 ACLSHMEM 操作后应调用一次。

    ```python
    def finalize() -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|-|无返回值。若销毁失败则引发 ``AclshmemError``|

1. 分配一个由 ACLSHMEM 支持的 NPU 缓冲区。这是一个集合（collective）操作，所有 PE 必须同步调用。

    ```python
    def buffer(size, release=False, except_on_del=True) -> Buffer
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |size|[in]|要分配的缓冲区大小（字节）|
    |release|[in]|预留参数，无实际意义|
    |except_on_del|[in]|预留参数，无实际意义|
    |返回值|[out]|通过地址和字节长度表示的原始内存缓冲区。若分配失败则引发 ``AclshmemError``|

1. 释放由 ``buffer()`` 分配的缓冲区。这是一个集合（collective）操作。

    ```python
    def free(buf: Buffer) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |buf|[in]|需要释放的缓冲区|
    |返回值|-|无返回值|

1. 获取指定 PE 上目标缓冲区的对称地址，可用于直接引用该 PE 上的数据。

    ```python
    def get_peer_buffer(buf: Buffer, pe: int) -> Buffer
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |buf|[in]|远程可访问数据的对称地址|
    |pe|[in]|PE 编号|
    |返回值|[out]|指定 PE 上的远程对称地址缓冲区。若获取失败则引发 ``AclshmemError``|

1. 从本地 PE 复制连续数据到指定 PE 的对称内存地址，并在完成后更新远程信号变量。当前仅支持 MTE（Memory Transfer Engine）。同步（blocking）接口。

    ```python
    def put_signal(dst: Buffer, src: Buffer, signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int=-1, stream=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 上目标数据的对称地址|
    |src|[in]|本地内存中的源数据地址|
    |signal_var|[in]|远程 PE 上待更新信号字的对称地址|
    |signal_val|[in]|用于更新信号变量的值|
    |signal_operation|[in]|信号变量更新操作。支持：``SignalOp.SIGNAL_SET`` / ``SignalOp.SIGNAL_ADD``|
    |remote_pe|[in]|远程 PE 编号，默认 -1|
    |stream|[in]|预留参数，忽略。底层使用默认流|
    |返回值|-|无返回值|

1. 在指定流上将本地 PE 的连续数据复制到远程 PE 的对称内存地址。调用者需同步流以确保传输完成。当前仅支持 MTE（Memory Transfer Engine）。非阻塞（non-blocking）接口。

    ```python
    def put(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 上目标数据的对称地址|
    |src|[in]|本地内存中的源数据地址|
    |remote_pe|[in]|远程 PE 编号，默认 -1|
    |stream|[in]|ACL 流对象，用于执行排序。传入 ``0`` 或 ``None`` 使用默认流|
    |返回值|-|无返回值|

1. 在指定流上将远程 PE 对称内存中的连续数据复制到本地缓冲区。调用者需同步流以确保传输完成。当前仅支持 MTE（Memory Transfer Engine）。非阻塞（non-blocking）接口。

    ```python
    def get(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|本地内存中的目标数据地址|
    |src|[in]|远程 PE 上源数据的对称地址|
    |remote_pe|[in]|远程 PE 编号，默认 -1|
    |stream|[in]|ACL 流对象，用于执行排序。传入 ``0`` 或 ``None`` 使用默认流|
    |返回值|-|无返回值|

1. 在指定的 PE 上对远程信号变量执行原子操作，操作在给定流上执行。调用者需同步流以观察结果。当前仅支持 MTE（Memory Transfer Engine）。非阻塞（non-blocking）接口。

    ```python
    def signal_op(signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int=-1, stream: int=None) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |signal_var|[in]|目标 PE 可访问的信号变量的本地地址|
    |signal_val|[in]|用于原子操作的值|
    |signal_operation|[in]|对远程信号执行的操作。支持：``SignalOp.SIGNAL_SET`` / ``SignalOp.SIGNAL_ADD``|
    |remote_pe|[in]|待更新远程信号变量所在的 PE 编号，默认 -1|
    |stream|[in]|ACL 流对象，用于执行排序。传入 ``None`` 将引发 ``AclshmemInvalid`` 异常|
    |返回值|-|无返回值。若 ``stream`` 为 ``None`` 则引发 ``AclshmemInvalid``|

1. 等待对称信号变量满足指定比较条件。等待操作在给定流上执行，调用在 host 侧立即返回。同步流后，条件 ``signal_var`` ``cmp`` ``signal_val`` 保证为真。当前仅支持 MTE（Memory Transfer Engine）。

    ```python
    def signal_wait(signal_var: Buffer, signal_val: int, signal_operation: ComparisonType, stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |signal_var|[in]|源信号变量的本地地址|
    |signal_val|[in]|与 signal_var 所指向值进行比较的值|
    |signal_operation|[in]|比较操作符。支持：``ComparisonType.CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |stream|[in]|ACL 流对象，用于执行排序。传入 ``None`` 将引发 ``AclshmemInvalid`` 异常|
    |返回值|-|无返回值。若 ``stream`` 为 ``None`` 则引发 ``AclshmemInvalid``|

1. 确保所有先前发出的对称数据操作在给定流上完成。quiet 操作排入指定流中，调用在 host 侧立即返回；调用者需同步流以观测完成。当前仅支持 MTE（Memory Transfer Engine）。

    ```python
    def quiet(stream: int) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |stream|[in]|执行 quiet 操作的 ACL 流。必须传入有效流，``None`` 将引发 ``AclshmemInvalid``|
    |返回值|-|无返回值。若 ``stream`` 为 ``None`` 则引发 ``AclshmemInvalid``|

1. 获取本地 PE 编号。

    ```python
    def my_pe() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|本地 PE 编号|

1. 获取当前进程在指定 team 中的 PE 编号。

    ```python
    def team_my_pe(team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|目标 team 的 ID|
    |返回值|[out]|指定 team 中的 PE 编号，若 team 无效则返回 -1|

1. 获取程序中运行的 PE 总数（world team 维度）。

    ```python
    def n_pes() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|PE 总数|

1. 获取指定 team 中的 PE 总数。

    ```python
    def team_n_pes(team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|目标 team 的 ID|
    |返回值|[out]|指定 team 中的 PE 数目，若 team 无效则返回 -1|

1. 查询共享内存模块的当前初始化状态。

    ```python
    def init_status() -> InitStatus
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|返回初始化状态枚举值：``NOT_INITIALIZED`` / ``SHM_CREATED`` / ``INITIALIZED`` / ``INVALID``|

##### 类
1. UniqueId 类 — 用于 UID 初始化的唯一标识符句柄。

    ```python
    class UniqueId:
        def __init__(self):
    ```

    |属性|方向|含义|
    |-|-|-|
    |version|[out]|版本信息|
    |my_rank|[out]|当前进程的 PE 编号|
    |n_pes|[out]|所有进程的 PE 总数|
    |internal|[out]|UID 的内部信息（字节）|

1. InitStatus 枚举类 — 共享内存模块的初始化状态。

    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```

    |枚举值|含义|
    |-|-|
    |NOT_INITIALIZED|未初始化|
    |SHM_CREATED|共享内存已创建|
    |INITIALIZED|初始化完成|
    |INVALID|无效状态|

1. SignalOp 枚举类 — 信号变量原子操作类型。

    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```

    |枚举值|含义|
    |-|-|
    |SIGNAL_SET|原子设置：将给定值写入远程信号|
    |SIGNAL_ADD|原子加：将给定值加到远程信号现有值上|

1. ComparisonType 枚举类 — 信号等待比较操作类型。

    ```python
    class ComparisonType(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

    |枚举值|含义|
    |-|-|
    |CMP_EQ|等于（==）|
    |CMP_NE|不等于（!=）|
    |CMP_GT|大于（>）|
    |CMP_GE|大于等于（>=）|
    |CMP_LT|小于（<）|
    |CMP_LE|小于等于（<=）|

### shmem._pyshmem API
##### 对外接口
1. 初始化共享内存模块。这是一个集合（collective）操作。

    ```python
    def aclshmem_init(mype, npes, mem_size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |mype|[in]|本地 PE 索引，范围 0 ~ npes-1|
    |npes|[in]|PE 总数|
    |mem_size|[in]|每个 PE 分配的内存大小（字节）|
    |返回值|[out]|成功返回 0，失败返回非零错误码|

1. 销毁共享内存模块，释放所有资源。

    ```python
    def aclshmem_finalize(instance_id=0) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |instance_id|[in]|实例 ID，默认 0|
    |返回值|-|无返回值|

1. 查询共享内存模块的当前初始化状态。

    ```python
    def aclshmemx_init_status() -> InitStatus
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|返回初始化状态枚举值。返回 ``INITIALIZED`` 表示初始化已完成|

1. 设置 TLS 私钥与密码，并注册解密回调函数。

    ```python
    def set_conf_store_tls_key(tls_pk, tls_pk_pw, py_decrypt_func:Callable[[str], str]) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |tls_pk|[in]|TLS 私钥内容|
    |tls_pk_pw|[in]|TLS 私钥密码内容|
    |py_decrypt_func|[in]|解密回调函数，接受 ``(str cipher_text)`` 返回 ``(str plain_text)``|
    |返回值|[out]|成功返回 0，失败返回非零错误码|

1. 分配对称内存。这是一个集合（collective）操作，内嵌隐式 barrier。

    ```python
    def aclshmem_malloc(size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |size|[in]|分配内存大小（字节）|
    |返回值|[out]|成功返回指向已分配内存的指针（int），若 size 为 0 或分配失败返回 0 并引发异常|

1. 分配零初始化的对称内存。集合操作，内嵌隐式 barrier。

    ```python
    def aclshmem_calloc(nmemb, size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |nmemb|[in]|元素数量|
    |size|[in]|每个元素的大小（字节）|
    |返回值|[out]|成功返回指向已分配内存的指针（int），若 nmemb 或 size 为 0 返回 0 并引发异常|

1. 分配指定对齐方式的对称内存。集合操作，内嵌隐式 barrier。

    ```python
    def aclshmem_align(alignment, size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |alignment|[in]|内存对齐要求（必须是 2 的幂）|
    |size|[in]|要分配的字节数|
    |返回值|[out]|成功返回指向已分配内存的指针（int），若分配失败则引发异常|

1. 释放由对称内存分配函数分配的内存空间。集合操作，内嵌隐式 barrier。

    ```python
    def aclshmem_free(ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |ptr|[in]|要释放的内存指针|
    |返回值|-|无返回值|

1. 获取指定 PE 上对称地址的远程引用地址，可用于直接访问该 PE 上的数据。

    ```python
    def aclshmem_ptr(ptr, peer) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |ptr|[in]|远程可访问数据的对称地址|
    |peer|[in]|PE 编号|
    |返回值|[out]|成功返回远程对称地址（int），若输入地址非法则返回 0|

1. 获取 PE 编号（world team 维度）。

    ```python
    def my_pe() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|本地 PE 编号|

1. 获取指定 team 中的 PE 编号。

    ```python
    def team_my_pe(team_id) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team_id|[in]|team 的句柄|
    |返回值|[out]|指定 team 中 PE 的编号，出错返回 -1|

1. 获取 PE 总数（world team 维度）。

    ```python
    def pe_count() -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|PE 总数|

1. 获取指定 team 中的 PE 数量。

    ```python
    def team_n_pes(team_id) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team_id|[in]|team 的句柄|
    |返回值|[out]|指定 team 中 PE 的数量，出错返回 -1|

1. 从现有父 team 中按步长拆分子 team。这是一个集合（collective）操作。

    ```python
    def team_split_strided(parent, start, stride, size) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |parent|[in]|父 team ID|
    |start|[in]|子 team 的起始 PE 编号|
    |stride|[in]|PE 编号之间的步长|
    |size|[in]|子 team 包含的 PE 数量|
    |返回值|[out]|成功返回新 team ID，出错返回 -1|

1. 基于二维笛卡尔空间从父 team 中拆分 team。集合操作。

    ```python
    def team_split_2d(parent, x_range) -> tuple
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |parent|[in]|父 team 句柄|
    |x_range|[in]|第一维度的元素数量|
    |返回值|[out]|成功返回 (x_team_id, y_team_id) 元组，失败引发异常|

1. 获取创建 team 时传入的 team 配置。

    ```python
    def aclshmem_team_get_config(team) -> TeamConfig
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|team ID|
    |返回值|[out]|成功返回 ``TeamConfig`` 对象，失败引发异常|

1. 将连续数据从本地 PE 复制到指定 PE 的对称地址。同步（blocking）接口。

    ```python
    def aclshmem_putmem(dst, src, elem_size, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将对称内存中指定 PE 上的连续数据复制到本地 PE。同步（blocking）接口。

    ```python
    def aclshmem_getmem(dst, src, elem_size, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将本地内存中按 sst 步距排列的数据复制到指定 PE 对称内存的 dst 步距位置。同步（blocking）接口。

    ```python
    def aclshmem_{TYPE}_iput(dest, source, dst, sst, nelems, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``double`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |dest|[in]|远程目标数据的对称内存指针|
    |source|[in]|本地源数据的指针|
    |dst|[in]|目标地址中连续元素之间的步长|
    |sst|[in]|源地址中连续元素之间的步长|
    |nelems|[in]|连续元素块的个数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将远程 PE 对称内存中按 sst 步距排列的数据复制到本地 dst 步距位置。同步（blocking）接口。

    ```python
    def aclshmem_{TYPE}_iget(dest, source, dst, sst, nelems, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``double`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |dest|[in]|本地目标数据的指针|
    |source|[in]|远程源数据的对称内存指针|
    |dst|[in]|目标地址中连续元素之间的步长|
    |sst|[in]|源地址中连续元素之间的步长|
    |nelems|[in]|连续元素块的个数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将连续数据从本地 PE 复制到指定 PE 的对称内存地址。同步（blocking）接口。

    ```python
    def aclshmem_put{BITS}(dst, src, elem_size, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将对称内存中指定 PE 上的连续数据复制到本地 PE。同步（blocking）接口。

    ```python
    def aclshmem_get{BITS}(dst, src, elem_size, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将本地内存中按 sst 步距排列的数据复制到指定 PE 的 dst 步距位置（位宽版本）。同步（blocking）接口。

    ```python
    def aclshmem_iput{BITS}(dest, source, dst, sst, nelems, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dest|[in]|远程目标数据的对称内存指针|
    |source|[in]|本地源数据的指针|
    |dst|[in]|目标地址中连续元素之间的步长|
    |sst|[in]|源地址中连续元素之间的步长|
    |nelems|[in]|连续元素块的个数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将远程 PE 对称内存中按 sst 步距排列的数据复制到本地 dst 步距位置（位宽版本）。同步（blocking）接口。

    ```python
    def aclshmem_iget{BITS}(dest, source, dst, sst, nelems, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dest|[in]|本地目标数据的指针|
    |source|[in]|远程源数据的对称内存指针|
    |dst|[in]|目标地址中连续元素之间的步长|
    |sst|[in]|源地址中连续元素之间的步长|
    |nelems|[in]|连续元素块的个数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 返回库的主版本号和次版本号。

    ```python
    def aclshmem_info_get_version() -> tuple
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|返回 (major, minor) 元组|

1. 返回供应商定义的名称字符串。

    ```python
    def aclshmem_info_get_name() -> str
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|供应商定义的名称字符串|

1. 将一个 team 中的给定 PE 编号转换为另一个 team 中的对应 PE 编号。

    ```python
    def team_translate_pe(src_team, src_pe, dest_team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |src_team|[in]|源 team ID|
    |src_pe|[in]|源 PE 编号|
    |dest_team|[in]|目标 team ID|
    |返回值|[out]|成功返回目标 team 中对应的 PE 编号，出错返回 -1|

1. 销毁一个 team。

    ```python
    def team_destroy(team) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|要销毁的 team ID|
    |返回值|-|无返回值|

1. 获取运行时 FFTS 配置。

    ```python
    def get_ffts_config() -> str
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|[out]|FFTS 配置字符串|

1. 将本地 PE 上的连续数据复制到指定 PE 的对称地址。异步（non-blocking）接口。

    ```python
    def aclshmem_putmem_nbi(dst, src, elem_size, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将对称内存中指定 PE 上的连续数据复制到本地 PE。异步（non-blocking）接口。

    ```python
    def aclshmem_getmem_nbi(dst, src, elem_size, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将连续数据从本地 PE 复制到指定 PE 的对称内存地址（位宽版本）。异步（non-blocking）接口。

    ```python
    def aclshmem_put{BITS}_nbi(dst, src, elem_size, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 将对称内存中指定 PE 上的连续数据复制到本地 PE（位宽版本）。异步（non-blocking）接口。

    ```python
    def aclshmem_get{BITS}_nbi(dst, src, elem_size, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 从本地 PE 复制连续数据到指定 PE 的对称地址，并在完成后更新远程信号变量。异步（non-blocking）接口。

    ```python
    def aclshmemx_putmem_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |sig|[in]|待更新信号字的对称地址|
    |signal|[in]|用于更新信号的值|
    |sig_op|[in]|信号更新操作。支持：``ACLSHMEM_SIGNAL_SET`` / ``ACLSHMEM_SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 从本地 PE 复制连续数据到指定 PE 的对称地址，并更新远程信号变量。同步（blocking）接口。

    ```python
    def aclshmemx_putmem_signal(dst, src, elem_size, sig, signal, sig_op, pe) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |sig|[in]|待更新信号字的对称地址|
    |signal|[in]|用于更新信号的值|
    |sig_op|[in]|信号更新操作。支持：``ACLSHMEM_SIGNAL_SET`` / ``ACLSHMEM_SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 从本地 PE 复制连续数据到指定 PE 对称地址并更新远程信号（位宽版本）。异步（non-blocking）接口。

    ```python
    def aclshmemx_put{BITS}_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |sig|[in]|待更新信号字的对称地址|
    |signal|[in]|用于更新信号的值|
    |sig_op|[in]|信号更新操作。支持：``ACLSHMEM_SIGNAL_SET`` / ``ACLSHMEM_SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 从本地 PE 复制连续数据到指定 PE 对称地址并更新远程信号（位宽版本）。同步（blocking）接口。

    ```python
    def aclshmemx_put{BITS}_signal(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |BITS|-|数据位宽：``8`` / ``16`` / ``32`` / ``64`` / ``128``|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |sig|[in]|待更新信号字的对称地址|
    |signal|[in]|用于更新信号的值|
    |sig_op|[in]|信号更新操作。支持：``ACLSHMEM_SIGNAL_SET`` / ``ACLSHMEM_SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |返回值|-|无返回值|

1. 所有 PE 通过广播调用 exit() 退出进程。

    ```python
    def aclshmem_global_exit(status) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |status|[in]|传递给 exit() 的状态值|
    |返回值|-|无返回值|

1. 确保所有先前发出的对称数据操作在默认流上完成。quiet 操作排入默认流中，调用在 host 侧立即返回；调用者需同步默认流以观测完成。

    ```python
    def aclshmem_quiet() -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |返回值|-|无返回值|

1. 获取指定 team 中的 PE 编号。

    ```python
    def my_pe(team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|team ID|
    |返回值|[out]|指定 team 中 PE 的编号，出错返回 -1|

1. 获取指定 team 中的 PE 数量。

    ```python
    def pe_count(team) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |team|[in]|team ID|
    |返回值|[out]|指定 team 中 PE 的数目，出错返回 -1|

1. 等待信号变量满足比较条件（``*sig_addr cmp cmp_val``）时返回。阻塞（blocking）接口。

    ```python
    def aclshmem_signal_wait_until(sig_addr, cmp, cmp_val) -> int
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |sig_addr|[in]|源信号变量的本地地址|
    |cmp|[in]|比较操作符。支持：``CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_val|[in]|比较值|
    |返回值|[out]|满足条件时 ``sig_addr`` 的值|

1. 等待单个元素满足比较条件 ``ivar cmp cmp_val`` 后返回（类型化版本）。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until(ivar, cmp, cmp_val) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivar|[in]|对称内存中信号变量的指针|
    |cmp|[in]|比较操作符|
    |cmp_val|[in]|比较值|
    |返回值|-|无返回值|

1. 等待信号变量不等于给定值时返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait(ivar, cmp_val) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivar|[in]|对称内存中信号变量的指针|
    |cmp_val|[in]|比较值|
    |返回值|-|无返回值|

1. 等待数组中所有元素均满足比较条件 ``ivars[i] cmp cmp_val`` 后返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_all(ivars_ptr, nelems, status_ptr, cmp, cmp_val) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_val|[in]|比较值|
    |返回值|-|无返回值|

1. 等待数组中至少有一个元素满足比较条件 ``ivars[i] cmp cmp_val`` 后返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_any(ivars_ptr, nelems, status_ptr, cmp, cmp_val, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_val|[in]|比较值|
    |res_out_ptr|[out]|接收满足比较条件的元素索引值|
    |返回值|-|无返回值|

1. 等待数组中至少有一个元素满足比较条件，并返回所有满足条件的元素索引。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_val, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |indices_ptr|[out]|接收满足条件元素的索引值数组|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_val|[in]|比较值|
    |res_out_ptr|[out]|接收满足比较条件的元素个数|
    |返回值|-|无返回值|

1. 等待数组中所有元素均满足向量比较条件 ``ivars[i] cmp cmp_values[i]`` 后返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_values_ptr|[in]|比较值数组|
    |返回值|-|无返回值|

1. 等待数组中至少有一个元素满足向量比较条件 ``ivars[i] cmp cmp_values[i]`` 后返回。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|接收满足比较条件的元素索引值|
    |返回值|-|无返回值|

1. 等待数组中至少有一个元素满足向量比较条件，并返回所有满足条件的元素索引。阻塞（blocking）接口。

    ```python
    def aclshmem_{TYPE}_wait_until_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |indices_ptr|[out]|接收满足条件元素的索引值数组|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|接收满足比较条件的元素个数|
    |返回值|-|无返回值|

1. 检查单个元素是否满足比较条件 ``ivar cmp cmp_value``。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test(ivar, cmp, cmp_value, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivar|[in]|对称内存中信号变量的指针|
    |cmp|[in]|比较操作符|
    |cmp_value|[in]|比较值|
    |res_out_ptr|[out]|满足条件返回 1，否则返回 0|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 返回|

1. 检查数组中是否至少有一个元素满足比较条件 ``ivars[i] cmp cmp_value``。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_any(ivars_ptr, nelems, status_ptr, cmp, cmp_value, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_value|[in]|比较值|
    |res_out_ptr|[out]|满足条件的元素索引值；若无元素满足或测试集为空则返回 ``SIZE_MAX``|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 返回|

1. 检查数组中是否至少有一个元素满足比较条件，并返回所有满足条件的元素索引。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_value, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |indices_ptr|[out]|接收满足条件元素的索引值数组|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_value|[in]|比较值|
    |res_out_ptr|[out]|满足条件的元素个数；若测试集为空则返回 0|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 和 ``indices_ptr`` 返回|

1. 检查数组中所有元素是否均满足向量比较条件 ``ivars[i] cmp cmp_values[i]``。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|全部满足或 nelems 为 0 返回 1，否则返回 0|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 返回|

1. 检查数组中是否至少有一个元素满足向量比较条件 ``ivars[i] cmp cmp_values[i]``。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|满足条件的第一个元素索引值；若无元素满足或测试集为空则返回 ``SIZE_MAX``|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 返回|

1. 检查数组中是否至少有一个元素满足向量比较条件，并返回所有满足条件的元素索引。非阻塞（non-blocking）查询接口。

    ```python
    def aclshmem_{TYPE}_test_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |TYPE|-|数据类型：``float`` / ``int8`` / ``int16`` / ``int32`` / ``int64`` / ``uint8`` / ``uint16`` / ``uint32`` / ``uint64`` / ``char``|
    |ivars_ptr|[in]|对称内存中长度为 ``nelems`` 的数组|
    |nelems|[in]|数组元素个数|
    |indices_ptr|[out]|接收满足条件元素的索引值数组|
    |status_ptr|[in]|可选本地掩码数组，传入 0 表示不使用|
    |cmp|[in]|比较操作符|
    |cmp_values_ptr|[in]|比较值数组|
    |res_out_ptr|[out]|满足条件的元素个数；若测试集为空则返回 0|
    |返回值|-|无返回值，结果通过 ``res_out_ptr`` 和 ``indices_ptr`` 返回|

1. 在指定流上将连续数据从本地 PE 复制到指定 PE 的对称地址。非阻塞（non-blocking）接口。

    ```python
    def aclshmemx_putmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|远程 PE 对称地址的指针|
    |src|[in]|本地源数据内存的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |stream|[in]|ACL 流对象。传入 ``0`` 或 ``None`` 使用默认流|
    |返回值|-|无返回值|

1. 在指定流上将对称内存中指定 PE 上的连续数据复制到本地 PE。非阻塞（non-blocking）接口。

    ```python
    def aclshmemx_getmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |dst|[in]|本地目标内存的指针|
    |src|[in]|远程 PE 对称地址的指针|
    |elem_size|[in]|目标地址和源地址中元素的总字节数|
    |pe|[in]|远程 PE 编号|
    |stream|[in]|ACL 流对象。传入 ``0`` 或 ``None`` 使用默认流|
    |返回值|-|无返回值|

1. 在指定流上对远程信号变量执行原子操作。当前仅支持 MTE。非阻塞（non-blocking）接口。

    ```python
    def aclshmemx_signal_op_on_stream(sig, signal, sig_op, pe, stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |sig|[in]|目标 PE 上可访问的信号变量的本地地址|
    |signal|[in]|用于原子操作的值|
    |sig_op|[in]|对远程信号执行的操作。支持：``ACLSHMEM_SIGNAL_SET`` / ``ACLSHMEM_SIGNAL_ADD``|
    |pe|[in]|远程 PE 编号|
    |stream|[in]|ACL 流对象。传入 ``0`` 或 ``None`` 使用默认流|
    |返回值|-|无返回值|

1. 非阻塞（non-blocking）接口。在指定流上等待信号变量满足比较条件。调用在 host 侧立即返回，等待在流上执行。调用者需同步流以确保等待完成。当前仅支持 MTE。

    ```python
    def aclshmemx_signal_wait_until_on_stream(sig, cmp, cmp_val, stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |sig|[in]|源信号变量的本地地址|
    |cmp|[in]|比较操作符。支持：``ACLSHMEM_CMP_EQ`` / ``CMP_NE`` / ``CMP_GT`` / ``CMP_GE`` / ``CMP_LT`` / ``CMP_LE``|
    |cmp_val|[in]|与 sig 所指向值进行比较的值|
    |stream|[in]|ACL 流对象，用于执行排序。必须传入有效流|
    |返回值|-|无返回值|

1. 确保所有先前发出的对称数据操作在给定流上完成。quiet 操作排入指定流中，调用在 host 侧立即返回；调用者需同步流以观测完成。当前仅支持 MTE。

    ```python
    def aclshmemx_quiet_on_stream(stream) -> None
    ```

    |参数/返回值|方向|含义|
    |-|-|-|
    |stream|[in]|执行 quiet 操作的 ACL 流。传入 ``0`` 或 ``None`` 使用默认流|
    |返回值|-|无返回值|

##### 类
1. OpEngineType 枚举类 — 数据传输引擎类型。

    ```python
    class OpEngineType(Enum):
        MTE
        SDMA
        ROCE
        UDMA
    ```

    |枚举值|含义|
    |-|-|
    |MTE|Memory Transfer Engine|
    |SDMA|System DMA|
    |ROCE|RDMA over Converged Ethernet|
    |UDMA|Unified DMA|

1. OptionalAttr 类 — 初始化可选属性配置。

    ```python
    class OptionalAttr:
        def __init__(self):
    ```

    |属性|方向|含义|
    |-|-|-|
    |version|[in]|配置版本号|
    |data_op_engine_type|[in]|数据传输引擎类型（``OpEngineType`` 枚举值）|
    |shm_init_timeout|[in]|init 操作的超时时间|
    |shm_create_timeout|[in]|create 操作的超时时间|
    |control_operation_timeout|[in]|控制操作的超时时间|
    |sockFd|[in]|socket 文件描述符，默认 -1|

1. InitAttr 类 — 初始化属性配置。

    ```python
    class InitAttr:
        def __init__(self):
    ```

    |属性|方向|含义|
    |-|-|-|
    |my_rank|[in]|当前进程的 PE 编号|
    |n_ranks|[in]|PE 总数|
    |ip_port|[in]|通信服务器的 IP 和端口|
    |local_mem_size|[in]|当前 PE 分配的对称内存大小（字节）|
    |option_attr|[in]|``OptionalAttr`` 可选属性配置|

1. TeamConfig 类 — Team 配置。

    ```python
    class TeamConfig:
    ```

    |属性|方向|含义|
    |-|-|-|
    |num_contexts|[in]|一个 team 中可以同时运行的上下文数量|

1. UniqueId 类 — UID 初始化的唯一标识符句柄。

    ```python
    class UniqueId:
        def __init__(self):
    ```

    |属性|方向|含义|
    |-|-|-|
    |version|[out]|版本信息|
    |my_pe|[out]|当前进程的 PE 编号|
    |n_pes|[out]|所有进程的 PE 总数|
    |internal|[out]|UID 的内部信息（字节）|

1. InitStatus 枚举类 — 共享内存模块初始化状态。

    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```

    |枚举值|含义|
    |-|-|
    |NOT_INITIALIZED|未初始化|
    |SHM_CREATED|共享内存已创建|
    |INITIALIZED|初始化完成|
    |INVALID|无效状态|

1. SignalOp 枚举类 — 信号变量原子操作类型。

    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```

    |枚举值|含义|
    |-|-|
    |SIGNAL_SET|原子设置：将给定值写入远程信号|
    |SIGNAL_ADD|原子加：将给定值加到远程信号现有值上|

1. CmpOp 枚举类 — 信号比较操作类型。

    ```python
    class CmpOp(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

    |枚举值|含义|
    |-|-|
    |CMP_EQ|等于（==）|
    |CMP_NE|不等于（!=）|
    |CMP_GT|大于（>）|
    |CMP_GE|大于等于（>=）|
    |CMP_LT|小于（<）|
    |CMP_LE|小于等于（<=）|