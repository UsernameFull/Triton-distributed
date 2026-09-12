/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>
#include <iostream>

#include "shmem.h"
#include "host_device/shmem_common_types.h"

namespace py = pybind11;

namespace shm {
namespace {

static py::function g_py_decrypt_func;
static py::function g_py_logger_func;
static constexpr size_t MAX_CIPHER_LEN(10 * 1024 * 1024);

int aclshmem_initialize(aclshmemx_init_attr_t &attributes)
{
    aclshmemx_bootstrap_t bootstrap_flags = ACLSHMEMX_INIT_WITH_DEFAULT;
    auto ret = aclshmemx_init_attr(bootstrap_flags, &attributes);
    if (ret != 0) {
        std::cerr << "initialize shmem failed, ret: " << ret << std::endl;
        return ret;
    }

    return 0;
}

py::bytes aclshmem_get_unique_id()
{
    aclshmemx_uniqueid_t uid;
    auto ret = aclshmemx_get_uniqueid(&uid);
    if (ret != 0) {
        throw std::runtime_error("get unique id failed");
    }
    return py::bytes(reinterpret_cast<const char*>(&uid), sizeof(uid));
}

int aclshmem_initialize_unique_id(int rank, int world_size, int64_t mem_size, const std::string &bytes)
{
    if (bytes.size() < sizeof(aclshmemx_uniqueid_t)) {
        std::cerr << "Error: Input bytes size (" << bytes.size()
                  << ") is smaller than required size (" << sizeof(aclshmemx_uniqueid_t)
                  << ")." << std::endl;
        return -1;
    }
    aclshmemx_uniqueid_t uid;
    std::copy_n(bytes.data(), sizeof(uid), reinterpret_cast<char*>(&uid));
    aclshmemx_init_attr_t attr;
    auto ret = aclshmemx_set_attr_uniqueid_args(rank, world_size, mem_size, &uid, &attr);
    if (ret != 0) {
        std::cerr << "set attr failed " << ret << std::endl;
        return ret;
    }
    aclshmemx_bootstrap_t bootstrap_flags = ACLSHMEMX_INIT_WITH_UNIQUEID;
    return aclshmemx_init_attr(bootstrap_flags, &attr);
}

static int py_decrypt_handler_wrapper(const char *cipherText, size_t cipherTextLen, char *plainText,
                                      size_t &plainTextLen)
{
    if (cipherTextLen > MAX_CIPHER_LEN || !g_py_decrypt_func || g_py_decrypt_func.is_none()) {
        std::cerr << "input cipher len is too long or decrypt func invalid." << std::endl;
        return -1;
    }

    try {
        py::str py_cipher = py::str(cipherText, cipherTextLen);
        std::string plain = py::cast<std::string>(g_py_decrypt_func(py_cipher).cast<py::str>());
        if (plain.size() >= plainTextLen) {
            std::cerr << "output cipher len is too long" << std::endl;
            std::fill(plain.begin(), plain.end(), 0);
            return -1;
        }

        std::copy(plain.begin(), plain.end(), plainText);
        plainText[plain.size()] = '\0';
        plainTextLen = plain.size();
        std::fill(plain.begin(), plain.end(), 0);
        return 0;
    } catch (const py::error_already_set &e) {
        return -1;
    }
}

int32_t aclshmem_set_conf_store_tls_key_with_decrypt(std::string &tls_pk, std::string &tls_pk_pw,
    std::optional<py::function> py_decrypt_func = std::nullopt)
{
    if (!py_decrypt_func || !py_decrypt_func.has_value()) {
        return aclshmemx_set_config_store_tls_key(tls_pk.c_str(), tls_pk.size(), tls_pk_pw.c_str(),
            tls_pk_pw.size(), nullptr);
    }

    g_py_decrypt_func = *py_decrypt_func;
    return aclshmemx_set_config_store_tls_key(tls_pk.c_str(), tls_pk.size(), tls_pk_pw.c_str(),
        tls_pk_pw.size(), py_decrypt_handler_wrapper);
}

static void bridge_logger(int level, const char *msg)
{
    if (g_py_logger_func) {
        py::gil_scoped_acquire acquire;
        g_py_logger_func(level, msg);
    }
}

int32_t aclshmem_set_extern_logger_py(py::function pyfunc)
{
    g_py_logger_func = pyfunc;
    return aclshmemx_set_extern_logger(&bridge_logger);
}

int32_t aclshmem_set_conf_store_tls_info(bool enable, std::string &tls_info)
{
    return aclshmemx_set_conf_store_tls(enable, tls_info.c_str(), tls_info.size());
}
}  // namespace
}  // namespace shm

void DefineShmemAttr(py::module_ &m)
{
    py::enum_<data_op_engine_type_t>(m, "OpEngineType")
        .value("MTE", ACLSHMEM_DATA_OP_MTE)
        .value("SDMA", ACLSHMEM_DATA_OP_SDMA)
        .value("ROCE", ACLSHMEM_DATA_OP_ROCE)
        .value("UDMA", ACLSHMEM_DATA_OP_UDMA);

    py::class_<aclshmem_init_optional_attr_t>(m, "OptionalAttr")
        .def(py::init([]() {
            auto optional_attr = new (std::nothrow) aclshmem_init_optional_attr_t;
            optional_attr->shm_init_timeout = DEFAULT_TIMEOUT;
            optional_attr->shm_create_timeout = DEFAULT_TIMEOUT;
            optional_attr->control_operation_timeout = DEFAULT_TIMEOUT;
            optional_attr->sockFd = -1;
            return optional_attr;
        }))
        .def_readwrite("version", &aclshmem_init_optional_attr_t::version)
        .def_readwrite("data_op_engine_type", &aclshmem_init_optional_attr_t::data_op_engine_type)
        .def_readwrite("shm_init_timeout", &aclshmem_init_optional_attr_t::shm_init_timeout)
        .def_readwrite("shm_create_timeout", &aclshmem_init_optional_attr_t::shm_create_timeout)
        .def_readwrite("control_operation_timeout", &aclshmem_init_optional_attr_t::control_operation_timeout)
        .def_readwrite("sockFd", &aclshmem_init_optional_attr_t::sockFd);

    py::class_<aclshmemx_init_attr_t>(m, "InitAttr")
        .def(py::init([]() {
            auto init_attr = new (std::nothrow) aclshmemx_init_attr_t;
            return init_attr;
        }))
        .def_readwrite("my_rank", &aclshmemx_init_attr_t::my_pe)
        .def_readwrite("n_ranks", &aclshmemx_init_attr_t::n_pes)
        .def_property("ip_port",
                      [](const aclshmemx_init_attr_t &self) {
                            return std::string(self.ip_port[0] != '\0' ? self.ip_port : "");
                        },
                      [](aclshmemx_init_attr_t &self, const std::string &value) {
                            size_t copy_len = std::min(value.size(), sizeof(self.ip_port) - 1);
                            std::copy_n(value.c_str(), copy_len, self.ip_port);
                            self.ip_port[copy_len] = '\0';
                        })
        .def_readwrite("local_mem_size", &aclshmemx_init_attr_t::local_mem_size)
        .def_readwrite("option_attr", &aclshmemx_init_attr_t::option_attr);

    py::class_<aclshmem_team_config_t>(m, "TeamConfig")
        .def(py::init<>())
        .def(py::init<int>())
        .def_readwrite("num_contexts", &aclshmem_team_config_t::num_contexts);
}

void DefineShmemInitStatus(py::module_ &m)
{
    py::enum_<aclshmemx_init_status_t>(m, "InitStatus")
        .value("NOT_INITIALIZED", ACLSHMEM_STATUS_NOT_INITIALIZED)
        .value("SHM_CREATED", ACLSHMEM_STATUS_SHM_CREATED)
        .value("INITIALIZED", ACLSHMEM_STATUS_IS_INITIALIZED)
        .value("INVALID", ACLSHMEM_STATUS_INVALID);
}

void DefineShmemUniqueId(py::module_ &m)
{
    constexpr int32_t MAGIC = (1 << 16) + sizeof(aclshmemx_uniqueid_t);

    py::class_<aclshmemx_uniqueid_t>(m, "UniqueId")
        .def(py::init([]() {
            aclshmemx_uniqueid_t obj{};
            obj.version = MAGIC;
            return obj;
        }))
        .def_readwrite("version", &aclshmemx_uniqueid_t::version)
        .def_readwrite("my_pe",   &aclshmemx_uniqueid_t::my_pe)
        .def_readwrite("n_pes",   &aclshmemx_uniqueid_t::n_pes)
        .def_property("internal",
            // getter: return bytes
            [](const aclshmemx_uniqueid_t& self) {
                return py::bytes(self.internal, ACLSHMEM_UNIQUE_ID_INNER_LEN);
            },
            // setter: accept bytes or bytearray
            [MAGIC](aclshmemx_uniqueid_t& self, const py::bytes& value) {
                if (self.version != MAGIC) {
                    throw std::invalid_argument(
                        "Cannot set 'internal': UniqueId is not initialized or has invalid magic number. "
                    );
                }
                auto buffer = value;
                char* ptr = nullptr;
                ssize_t size = 0;
                if (PYBIND11_BYTES_AS_STRING_AND_SIZE(buffer.ptr(), &ptr, &size)) {
                    throw std::runtime_error("Failed to extract bytes");
                }
                if (size != ACLSHMEM_UNIQUE_ID_INNER_LEN) {
                    throw std::invalid_argument(
                        "internal must be exactly " + std::to_string(ACLSHMEM_UNIQUE_ID_INNER_LEN) + " bytes");
                }
                std::memcpy(self.internal, ptr, ACLSHMEM_UNIQUE_ID_INNER_LEN);
            });
}

void DefineShmemSignalOp(py::module_ &m)
{
    py::enum_<aclshmem_signal_op_type_t>(m, "SignalOp")
        .value("SIGNAL_SET", ACLSHMEM_SIGNAL_SET)
        .value("SIGNAL_ADD", ACLSHMEM_SIGNAL_ADD);
}

void DefineShmemCmpOp(py::module_ &m)
{
    py::enum_<aclshmem_cmp_op_type_t>(m, "CmpOp")
        .value("CMP_EQ", ACLSHMEM_CMP_EQ)
        .value("CMP_NE", ACLSHMEM_CMP_NE)
        .value("CMP_GT", ACLSHMEM_CMP_GT)
        .value("CMP_GE", ACLSHMEM_CMP_GE)
        .value("CMP_LT", ACLSHMEM_CMP_LT)
        .value("CMP_LE", ACLSHMEM_CMP_LE);
}

void DefineShmemMemType(py::module_ &m)
{
    py::enum_<aclshmem_mem_type_t>(m, "MemType")
        .value("HOST_SIDE", HOST_SIDE)
        .value("DEVICE_SIDE", DEVICE_SIDE);
}

PYBIND11_MODULE(_pyshmem, m)
{
    DefineShmemAttr(m);
    DefineShmemInitStatus(m);
    DefineShmemUniqueId(m);
    DefineShmemSignalOp(m);
    DefineShmemCmpOp(m);
    DefineShmemMemType(m);

    m.def("aclshmem_init", &shm::aclshmem_initialize, py::call_guard<py::gil_scoped_release>(), py::arg("attributes"), R"(
Initialize share memory module.

Arguments:
    attributes(InitAttr): attributes used to init shmem.
Returns:
    returns zero on success. On error, -1 is returned.
    )");

    m.def("aclshmem_finalize", &aclshmem_finalize, py::call_guard<py::gil_scoped_release>(),
          py::arg("instance_id") = 0,
          R"(
Finalize share memory module.

Arguments:
    instance_id(uint64_t): Instance ID. Default is 0.
    )");


    m.def("aclshmem_init_using_unique_id", &shm::aclshmem_initialize_unique_id,
          py::call_guard<py::gil_scoped_release>(), py::arg("mype"),
          py::arg("npes"), py::arg("mem_size"), py::arg("uid"), R"(
Initialize share memory module using unique id. This function need run with PTA.

Arguments:
    mype(int): local processing element index, range in [0, npes).
    npes(int): total count of processing elements.
    mem_size(int): memory size for each processing element in bytes.
    uid(aclshmem_uid): shmem uid.
Returns:
    returns zero on success. On error, -1 is returned.
    )");

    m.def("aclshmem_get_unique_id", &shm::aclshmem_get_unique_id, py::call_guard<py::gil_scoped_release>(), R"(
Get the unique id. This function need run with PTA.

Returns:
    returns unique id on success. On error, raise error.
    )");

    m.def(
        "aclshmemx_init_status",
        []() {
            int32_t ret = aclshmemx_init_status();
            return static_cast<aclshmemx_init_status_t>(ret);
        },
        py::call_guard<py::gil_scoped_release>(), R"(
Query the current initialization status of shared memory module.

Returns:
    Returns initialization status. Returning ACLSHMEM_STATUS_IS_INITIALIZED indicates that initialization is complete.
    All return types can be found in aclshmemx_init_status_t.
    )");

    m.def("set_conf_store_tls_key", &shm::aclshmem_set_conf_store_tls_key_with_decrypt,
          py::call_guard<py::gil_scoped_release>(), py::arg("tls_pk"), py::arg("tls_pk_pw"),
          py::arg("py_decrypt_func").none(), R"(
Set the TLS private key and password, and register a decrypt key password handler.
Parameters:
    tls_pk (string): the content of tls private key string
    tls_pk_pw (string): the content of tls private key password string
    py_decrypt_func (callable): Python function that accepts (str cipher_text) and returns (str plain_text)
        cipher_text: the encrypted text (private key password)
        plain_text: the decrypted text (private key password)
Returns:
    returns zero on success. On error, none-zero is returned.
)");

    m.def("set_conf_store_tls", &shm::aclshmem_set_conf_store_tls_info, py::call_guard<py::gil_scoped_release>(),
          py::arg("enable"), py::arg("tls_info"), R"(
Set the config store tls info.
Parameters:
    enable (boolean): whether to enabel config store tls
        tls_info (string): tls config string
Returns:
    returns zero on success. On error, none-zero is returned.
)");

    m.def(
        "aclshmem_malloc",
        [](size_t size) {
            auto ptr = aclshmem_malloc(size);
            if (ptr == nullptr) {
                throw std::runtime_error("aclshmem_malloc failed");
            }
            return (intptr_t)ptr;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("size"),
        R"(
Allocates size bytes and returns a pointer to the allocated memory. The memory is not initialized. If size is 0, then
aclshmem_malloc() returns NULL.
    )");

    m.def(
        "aclshmem_calloc",
        [](size_t nmemb, size_t size) {
            auto ptr = aclshmem_calloc(nmemb, size);
            if (ptr == nullptr) {
                throw std::runtime_error("aclshmem_calloc failed");
            }
            return (intptr_t)ptr;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("nmemb"), py::arg("size"),
        R"(
Allocates memory for an array of nmemb elements of size bytes each and returns a pointer to the allocated memory.
The memory is set to zero. If nmemb or size is 0, then returns NULL.

Arguments:
    nmemb(int): number of elements
    size(int): size of each element in bytes
Returns:
    pointer to the allocated memory
    )");

    m.def(
        "aclshmem_align",
        [](size_t alignment, size_t size) {
            auto ptr = aclshmem_align(alignment, size);
            if (ptr == nullptr) {
                throw std::runtime_error("aclshmem_align failed");
            }
            return (intptr_t)ptr;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("alignment"), py::arg("size"),
        R"(
Allocates size bytes of memory with specified alignment and returns a pointer to the allocated memory.
The memory address will be a multiple of the given alignment value (must be a power of two).

Arguments:
    alignment(int): required memory alignment (must be power of two)
    size(int): number of bytes to allocate
Returns:
    Pointer to the allocated memory, or NULL if allocation failed
    )");

    m.def(
        "aclshmem_free",
        [](intptr_t ptr) {
            auto mem = (void *)ptr;
            aclshmem_free(mem);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("ptr"),
        R"(
Frees the memory space pointed to by ptr, which must have been returned by a previous call to aclshmem_malloc.
    )");

    m.def(
        "aclshmem_ptr", [](intptr_t ptr, int pe) { return (intptr_t)aclshmem_ptr((void *)ptr, pe); },
        py::call_guard<py::gil_scoped_release>(), py::arg("ptr"), py::arg("peer"), R"(
Get address that may be used to directly reference dest on the specified PE.

Arguments:
    ptr(int): The symmetric address of the remotely accessible data.
    pe(int): PE number
    )");

    m.def(
        "aclshmemx_get_heap_base",
        [](py::object mem_type) {
            aclshmem_mem_type_t mt = mem_type.is_none() ? DEVICE_SIDE : mem_type.cast<aclshmem_mem_type_t>();
            return (intptr_t)aclshmemx_get_heap_base(mt);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("mem_type") = py::none(),
        R"(
Returns the start address (heap_base) of the local symmetric memory heap.

Arguments:
    mem_type(MemType): Allocation location of symmetric memory (MemType.HOST_SIDE / MemType.DEVICE_SIDE), default is MemType.DEVICE_SIDE
Returns:
    Pointer to the start address of the symmetric memory heap, or 0 if not initialized.
    )");

    m.def("my_pe", &aclshmem_my_pe, py::call_guard<py::gil_scoped_release>(), R"(Get my PE number.)");

    m.def(
        "team_my_pe", [](int team_id) { return aclshmem_team_my_pe(team_id); }, py::call_guard<py::gil_scoped_release>(),
        py::arg("team_id"),
        R"(
Get my PE number in specific team.
    )");

    m.def("pe_count", &aclshmem_n_pes, py::call_guard<py::gil_scoped_release>(), R"(Get number of PEs.)");

    m.def(
        "team_n_pes", [](int team_id) { return aclshmem_team_n_pes(team_id); }, py::call_guard<py::gil_scoped_release>(),
        py::arg("team_id"),
        R"(
Get number of PEs in specific team.
    )");

    m.def(
        "team_split_strided",
        [](int parent, int start, int stride, int size) {
            aclshmem_team_t new_team;
            auto ret = aclshmem_team_split_strided(parent, start, stride, size, &new_team);
            if (ret != 0) {
                std::cerr << "split parent team(" << parent << ") failed: " << ret << std::endl;
                return ret;
            }
            return new_team;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("parent"), py::arg("start"), py::arg("stride"),
        py::arg("size"), R"(
Split team from an existing parent team, this is a collective operation.

Arguments:
    parent(int): parent team id
    start(int): the lowest PE number of the subset of PEs from parent team that will form the new team
    stride(int): the stride between team PE numbers in the parent team
    size(int): the number of PEs from the parent team
Returns:
    On success, returns new team id. On error, -1 is returned.
    )");

    m.def(
        "team_split_2d",
        [](int parent, int x_range) {
            aclshmem_team_t new_x_team;
            aclshmem_team_t new_y_team;
            auto ret = aclshmem_team_split_2d(parent, x_range, &new_x_team, &new_y_team);
            if (ret != 0) {
                throw std::runtime_error("aclshmem_team_split_2d failed");
            }
            return std::make_pair(new_x_team, new_y_team);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("parent"), py::arg("x_range"), R"(
Collective Interface. Split team from an existing parent team based on a 2D Cartsian Space

Arguments:
    parent_team       [in] A team handle.
    x_range           [in] represents the number of elements in the first dimensions
Returns:
    On success, returns new x team id and new y team id. On error, raise error.
    )");

    m.def(
        "aclshmem_team_get_config",
        [](int team) {
            aclshmem_team_config_t team_config;
            auto ret = aclshmem_team_get_config(team, &team_config);
            if (ret != 0) {
                throw std::runtime_error("get team config failed");
            }
            return team_config;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Return team config which pass in as team created

Arguments:
    team(int)                 [in] team id
Returns:
    0 if success, -1 if fail.
    )");

    m.def(
        "aclshmem_putmem",
        [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {
            auto dst_addr = (void *)dst;
            auto src_addr = (void *)src;
            aclshmem_putmem(dst_addr, src_addr, elem_size, pe);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),
        R"(
Synchronous interface. Copy contiguous data on symmetric memory from local PE to address on the specified PE

Arguments:
    dst                [in] Pointer on Symmetric address of remote PE.
    src                [in] Pointer on local memory of the source data.
    elem_size          [in] size of elements in the destination and source addr.
    pe                 [in] PE number of the remote PE.
    )");

#define PYBIND_ACLSHMEM_TYPE_IPUT(NAME, TYPE)                                                                               \
    {                                                                                                                       \
        std::string funcName = "aclshmem_" #NAME "_iput";                                                                   \
        m.def(                                                                                                              \
            funcName.c_str(),                                                                                               \
            [](intptr_t dest, intptr_t source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {                       \
                auto dst_addr = (TYPE *)dest;                                                                               \
                auto src_addr = (TYPE *)source;                                                                             \
                aclshmem_##NAME##_iput(dst_addr, src_addr, dst, sst, nelems, pe);                                           \
            },                                                                                                              \
            py::call_guard<py::gil_scoped_release>(), py::arg("dest"), py::arg("source"), py::arg("dst"), py::arg("sst"),   \
            py::arg("nelems"), py::arg("pe"), R"(                                                                           \
        Synchronous interface. Copy strided data elements (specified by sst) of an array from a source array on the         \
        local PE to locations specified by stride dst on a dest array on specified remote PE.                               \
        Arguments:                                                                                                          \
            dest               [in] Pointer on Symmetric address of the destination data.
            source             [in] Pointer on local memory of the source data.
            dst                [in] The stride between consecutive elements of the dest array.
            sst                [in] The stride between consecutive elements of the source array.
            nelems             [in] Number of elements in the destination and source arrays.
            pe                 [in] PE number of the remote PE.
            )");                                                                                                            \
    }

ACLSHMEM_TYPE_FUNC(PYBIND_ACLSHMEM_TYPE_IPUT)
#undef PYBIND_ACLSHMEM_TYPE_IPUT

#define PYBIND_ACLSHMEM_PUT_SIZE(BITS)                                                                         		  		\
    {                                                                                                                 		\
        std::string funcName = "aclshmem_put" #BITS "";                                                        		  		\
        m.def(                                                                                                        		\
            funcName.c_str(),                                                                                         		\
            [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {         										  		\
                auto dst_addr = (void *)dst;                                                                          		\
                auto src_addr = (void *)src;                                                                          		\
                aclshmem_put##BITS(dst_addr, src_addr, elem_size, pe);                								  		\
            },                                                                                                        		\
            py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),  \
            R"(                                  																		    \
        Synchronous interface. Copy contiguous data on symmetric memory from local PE to address on the specified PE.
                                                                                                                            \
        Arguments:                                                                                                    		\
            dst                [in] Pointer on Symmetric address of the remote PE.
            src                [in] Pointer on local memory of the source data.
            elem_size          [in] size of elements in the destination and source addr.
            pe                 [in] PE number of the remote PE.
            )");                                                                                                      		\
    }

ACLSHMEM_SIZE_FUNC(PYBIND_ACLSHMEM_PUT_SIZE)
#undef PYBIND_ACLSHMEM_PUT_SIZE

#define PYBIND_ACLSHMEM_IPUT_SIZE(BITS)                                                                                     \
    {                                                                                                                       \
        std::string funcName = "aclshmem_iput" #BITS "";                                                                    \
        m.def(                                                                                                              \
            funcName.c_str(),                                                                                               \
            [](intptr_t dest, intptr_t source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {                       \
                auto dst_addr = (void *)dest;                                                                               \
                auto src_addr = (void *)source;                                                                             \
                aclshmem_iput##BITS(dst_addr, src_addr, dst, sst, nelems, pe);                                              \
            },                                                                                                              \
            py::call_guard<py::gil_scoped_release>(), py::arg("dest"), py::arg("source"), py::arg("dst"), py::arg("sst"),   \
            py::arg("nelems"), py::arg("pe"), R"(                                                                           \
        Synchronous interface. Copy strided data elements (specified by sst) of an array from a source array on the         \
        local PE to locations specified by stride dst on a dest array on specified remote PE.                               \
        Arguments:                                                                                                          \
            dest               [in] Pointer on symmetric address of the destination data.
            source             [in] Pointer on local memory of the source data.
            dst                [in] The stride between consecutive elements of the dest array.
            sst                [in] The stride between consecutive elements of the source array.
            nelems             [in] Number of elements in the destination and source arrays.
            pe                 [in] PE number of the remote PE.
            )");                                                                                                            \
    }

ACLSHMEM_SIZE_FUNC(PYBIND_ACLSHMEM_IPUT_SIZE)
#undef PYBIND_ACLSHMEM_IPUT_SIZE

    m.def(
        "aclshmem_getmem",
        [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {
            auto dst_addr = (void *)dst;
            auto src_addr = (void *)src;
            aclshmem_getmem(dst_addr, src_addr, elem_size, pe);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),
        R"(
Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE

Arguments:
    dst                [in] Pointer on local memory of the local PE.
    src                [in] Pointer on symmetric address of the source data.
    elem_size          [in] size of elements in the destination and source addr.
    pe                 [in] PE number of the remote PE.
    )");

#define PYBIND_ACLSHMEM_TYPE_IGET(NAME, TYPE)                                                                               \
    {                                                                                                                       \
        std::string funcName = "aclshmem_" #NAME "_iget";                                                                   \
        m.def(                                                                                                              \
            funcName.c_str(),                                                                                               \
            [](intptr_t dest, intptr_t source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {                       \
                auto dst_addr = (TYPE *)dest;                                                                               \
                auto src_addr = (TYPE *)source;                                                                             \
                aclshmem_##NAME##_iget(dst_addr, src_addr, dst, sst, nelems, pe);                                           \
            },                                                                                                              \
            py::call_guard<py::gil_scoped_release>(), py::arg("dest"), py::arg("source"), py::arg("dst"), py::arg("sst"),   \
            py::arg("nelems"), py::arg("pe"), R"(                                                                           \
        Synchronous interface. Copy strided data elements from a symmetric array from a specified remote PE to              \
        strided locations on a local array.                                                                                 \
        Arguments:                                                                                                          \
            dest               [in] Pointer on local memory of the destination data.
            source             [in] Pointer on symmetric address of the source data.
            dst                [in] The stride between consecutive elements of the dest array.
            sst                [in] The stride between consecutive elements of the source array.
            nelems             [in] Number of elements in the destination and source arrays.
            pe                 [in] PE number of the remote PE.
            )");                                                                                                            \
    }

ACLSHMEM_TYPE_FUNC(PYBIND_ACLSHMEM_TYPE_IGET)
#undef PYBIND_ACLSHMEM_TYPE_IGET

#define PYBIND_ACLSHMEM_GET_SIZE(BITS)                                                                         		  		\
    {                                                                                                                 		\
        std::string funcName = "aclshmem_get" #BITS "";                                                        		  		\
        m.def(                                                                                                        		\
            funcName.c_str(),                                                                                         		\
            [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {         										  		\
                auto dst_addr = (void *)dst;                                                                          		\
                auto src_addr = (void *)src;                                                                          		\
                aclshmem_get##BITS(dst_addr, src_addr, elem_size, pe);                								  		\
            },                                                                                                        		\
            py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),  \
            R"(                                  																		    \
        Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
																															\
        Arguments:                                                                                                    		\
			dst                [in] Pointer on local memory of the destination data.
			src                [in] Pointer on symmetric address of the source data.
			elem_size          [in] size of elements in the destination and source addr.
            pe                 [in] PE number of the remote PE.
            )");                                                                                                      		\
    }

ACLSHMEM_SIZE_FUNC(PYBIND_ACLSHMEM_GET_SIZE)
#undef PYBIND_ACLSHMEM_GET_SIZE

#define PYBIND_ACLSHMEM_IGET_SIZE(BITS)                                                                                     \
    {                                                                                                                       \
        std::string funcName = "aclshmem_iget" #BITS "";                                                                    \
        m.def(                                                                                                              \
            funcName.c_str(),                                                                                               \
            [](intptr_t dest, intptr_t source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {                       \
                auto dst_addr = (void *)dest;                                                                               \
                auto src_addr = (void *)source;                                                                             \
                aclshmem_iget##BITS(dst_addr, src_addr, dst, sst, nelems, pe);                                              \
            },                                                                                                              \
            py::call_guard<py::gil_scoped_release>(), py::arg("dest"), py::arg("source"), py::arg("dst"), py::arg("sst"),   \
            py::arg("nelems"), py::arg("pe"), R"(                                                                           \
        Synchronous interface. Copy strided data elements from a symmetric array from a specified remote PE to              \
        strided locations on a local array.                                                                                 \
        Arguments:                                                                                                          \
            dest               [in] Pointer on local memory of the destination data.
            source             [in] Pointer on symmetric address of the source data.
            dst                [in] The stride between consecutive elements of the dest array.
            sst                [in] The stride between consecutive elements of the source array.
            nelems             [in] Number of elements in the destination and source arrays.
            pe                 [in] PE number of the remote PE.
            )");                                                                                                            \
    }

ACLSHMEM_SIZE_FUNC(PYBIND_ACLSHMEM_IGET_SIZE)
#undef PYBIND_ACLSHMEM_IGET_SIZE

    m.def(
        "aclshmem_info_get_version",
        []() {
            int major = 0;
            int minor = 0;
            aclshmem_info_get_version(&major, &minor);
            return std::make_pair(major, minor);
        },
        py::call_guard<py::gil_scoped_release>(), R"(
Returns the major and minor version.

Arguments:
    None
Returns:
    major(int)      [out]major version
    minor(int)      [out]minor version
    )");

    m.def(
        "aclshmem_info_get_name",
        []() {
            char name[ACLSHMEM_MAX_NAME_LEN];
            aclshmem_info_get_name(name);
            std::string ret_name(name);
            return ret_name;
        },
        py::call_guard<py::gil_scoped_release>(), R"(
returns the vendor defined name string.

Arguments:
    None
Returns:
    name(str)      [out]defined name
    )");

#define PYBIND_ACLSHMEM_TYPENAME_P(NAME, TYPE)                                                         \
    {                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_p";                                                 \
        m.def(                                                                                         \
            funcName.c_str(),                                                                          \
            [](intptr_t dst, const TYPE value, int pe) {                                               \
                auto dst_addr = (TYPE *)dst;                                                           \
                aclshmem_##NAME##_p(dst_addr, value, pe);                                              \
            },                                                                                         \
            py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("value"), py::arg("pe"), \
            R"(                                                                                        \
    Provide a low latency put capability for single element of most basic types                        \
                                                                                                       \
    Arguments:                                                                                         \
        dst     [in] Symmetric address of the destination data on remote PE.
        value   [in] The element to be put.
        pe      [in] The number of the remote PE.
        )");                                                                                           \
    }

    ACLSHMEM_TYPE_FUNC(PYBIND_ACLSHMEM_TYPENAME_P)
#undef PYBIND_ACLSHMEM_TYPENAME_P

#define PYBIND_ACLSHMEM_TYPENAME_G(NAME, TYPE)                                       \
    {                                                                                \
        std::string funcName = "aclshmem_" #NAME "_g";                               \
        m.def(                                                                       \
            funcName.c_str(),                                                        \
            [](intptr_t src, int pe) {                                               \
                auto src_addr = (TYPE *)src;                                         \
                return aclshmem_##NAME##_g(src_addr, pe);                            \
            },                                                                       \
            py::call_guard<py::gil_scoped_release>(), py::arg("src"), py::arg("pe"), \
            R"(                                                                      \
    Provide a low latency get single element of most basic types.
                                                                                     \
    Arguments:                                                                       \
        src     [in] Symmetric address of the source data on remote PE.
        pe      [in] The number of the remote PE.
        A single element of type specified in the input pointer.
        )");                                                                         \
    }

    ACLSHMEM_TYPE_FUNC(PYBIND_ACLSHMEM_TYPENAME_G)
#undef PYBIND_ACLSHMEM_TYPENAME_G

    m.def("team_translate_pe", &aclshmem_team_translate_pe, py::call_guard<py::gil_scoped_release>(), py::arg("src_team"),
          py::arg("src_pe"), py::arg("dest_team"), R"(
Translate a given PE number in one team into the corresponding PE number in another team

Arguments:
    src_team(int): source team id
    src_pe(int): source PE number
    dest_team(int): destination team id
Returns:
    On success, returns the specified PE's number in the dest_team. On error, -1 is returned.
    )");

    m.def("team_destroy", &aclshmem_team_destroy, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Destroy a team with team id

Arguments:
    team(int): team id to be destroyed
    )");

    m.def("get_ffts_config", &util_get_ffts_config, py::call_guard<py::gil_scoped_release>(), R"(
Get runtime ffts config. This config should be passed to MIX Kernel and set by MIX Kernel using aclshmemx_set_ffts.
    )");

    m.def(
        "aclshmem_putmem_nbi",
        [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {
            auto dst_addr = (void *)dst;
            auto src_addr = (void *)src;
            aclshmem_putmem_nbi(dst_addr, src_addr, elem_size, pe);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),
        R"(
    Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.

    Arguments:
        dst                [in] Pointer on symmetric address of the destination data on remote PE.
        src                [in] Pointer on local memory of the source data.
        elem_size          [in] size of elements in the destination and source addr.
        pe                 [in] PE number of the remote PE.
        )");

#define PYBIND_ACLSHMEM_PUT_SIZE_NBI(BITS)                                                                         		    \
    {                                                                                                                 		\
        std::string funcName = "aclshmem_put" #BITS "_nbi";                                                        		    \
        m.def(                                                                                                        		\
            funcName.c_str(),                                                                                         		\
            [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {         										  		\
                auto dst_addr = (void *)dst;                                                                          		\
                auto src_addr = (void *)src;                                                                          		\
                aclshmem_put##BITS##_nbi(dst_addr, src_addr, elem_size, pe);                								\
            },                                                                                                        		\
            py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),  \
            R"(                                  																		    \
        Asynchronous interface. Copy contiguous data on symmetric memory from local PE to address on the specified PE.
																															\
        Arguments:                                                                                                    		\
			dst                [in] Pointer on symmetric address of the destination data on remote PE.
			src                [in] Pointer on local memory of the source data.
			elem_size          [in] size of elements in the destination and source addr.
            pe                 [in] PE number of the remote PE.
            )");                                                                                                      		\
    }

ACLSHMEM_SIZE_FUNC(PYBIND_ACLSHMEM_PUT_SIZE_NBI)
#undef PYBIND_ACLSHMEM_PUT_SIZE_NBI

    m.def(
        "aclshmem_getmem_nbi",
        [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {
            auto dst_addr = (void *)dst;
            auto src_addr = (void *)src;
            aclshmem_getmem_nbi(dst_addr, src_addr, elem_size, pe);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),
        R"(
    Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE

    Arguments:
        dst                [in] Pointer on local memory of the destination data on local PE.
        src                [in] Pointer on symmetric address of the source data on remote PE.
        elem_size          [in] size of elements in the destination and source addr.
        pe                 [in] PE number of the remote PE.
        )");

#define PYBIND_ACLSHMEM_GET_SIZE_NBI(BITS)                                                                         		    \
    {                                                                                                                 		\
        std::string funcName = "aclshmem_get" #BITS "_nbi";                                                        		    \
        m.def(                                                                                                        		\
            funcName.c_str(),                                                                                         		\
            [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {         										  		\
                auto dst_addr = (void *)dst;                                                                          		\
                auto src_addr = (void *)src;                                                                          		\
                aclshmem_get##BITS##_nbi(dst_addr, src_addr, elem_size, pe);                								\
            },                                                                                                        		\
            py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),  \
            R"(                                  																		    \
        Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
																															\
        Arguments:                                                                                                    		\
			dst                [in] Pointer on local memory of the destination data on local PE.
			src                [in] Pointer on symmetric address of the source data on remote PE.
			elem_size          [in] size of elements in the destination and source addr.
            pe                 [in] PE number of the remote PE.
            )");                                                                                                      		\
    }

ACLSHMEM_SIZE_FUNC(PYBIND_ACLSHMEM_GET_SIZE_NBI)
#undef PYBIND_ACLSHMEM_GET_SIZE_NBI

    m.def(
        "aclshmemx_putmem_signal_nbi",
        [](intptr_t dst, intptr_t src, size_t elem_size, intptr_t sig, int32_t signal, int sig_op, int pe) {
            auto dst_addr = (void *)dst;
            auto src_addr = (void *)src;
            auto sig_addr = (void *)sig;
            aclshmemx_putmem_signal_nbi(dst_addr, src_addr, elem_size, sig_addr, signal, sig_op, pe);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("sig"),
        py::arg("signal"), py::arg("sig_op"), py::arg("pe"), R"(
    Asynchronous interface. Copy contiguous data from the address on the local PE to symmetric memory on the specified PE

    Arguments:
        dst                [in] Pointer on symmetric address of the destination data on remote PE.
        src                [in] Pointer on local memory of the source data on local PE.
        elem_size          [in] size of elements in the destination and source addr.
        sig                [in] Symmetric address of the signal word to be updated.
        signal             [in] The value used to update sig_addr.
        sig_op             [in] Operation used to update sig_addr with signal.
                                Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
        pe                 [in] PE number of the remote PE.
        )");

    m.def(
        "aclshmemx_putmem_signal",
        [](intptr_t dst, intptr_t src, size_t elem_size, intptr_t sig, int32_t signal, int sig_op, int pe) {
            auto dst_addr = (void *)dst;
            auto src_addr = (void *)src;
            auto sig_addr = (void *)sig;
            aclshmemx_putmem_signal(dst_addr, src_addr, elem_size, sig_addr, signal, sig_op, pe);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("sig"),
        py::arg("signal"), py::arg("sig_op"), py::arg("pe"), R"(
    Synchronous interface. Copy contiguous data on symmetric memory from the local PE to address on the specified PE

    Arguments:
        dst                [in] Pointer on symmetric address of the destination data on remote PE.
        src                [in] Pointer on local memory of the source data on local PE.
        elem_size          [in] size of elements in the destination and source addr.
        sig                [in] Symmetric address of the signal word to be updated.
        signal             [in] The value used to update sig_addr.
        sig_op             [in] Operation used to update sig_addr with signal.
                                Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
        pe                 [in] PE number of the remote PE.
        )");

    m.def(
        "aclshmemx_putmem_on_stream",
        [](intptr_t dst, intptr_t src, size_t elem_size, int pe, intptr_t stream) {
            auto dst_addr = (void *)dst;
            auto src_addr = (void *)src;
            aclrtStream acl_stream = nullptr;
            if (stream != 0) {
                acl_stream = reinterpret_cast<aclrtStream>(stream);
            }
            aclshmemx_putmem_on_stream(dst_addr, src_addr, elem_size, pe, acl_stream);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),
        py::arg("stream"), R"(
    Copy contiguous data from the local PE to a symmetric memory address on a remote PE.

    Arguments:
        dst                [in] Pointer on symmetric address of the destination data.
        src                [in] Pointer on local memory of the source data.
        elem_size          [in] Number of elements in the dest and source arrays.
        pe                 [in] PE number of the remote PE.
        stream             [in] copy used stream(use default stream if stream == NULL).
        )");

    m.def(
        "aclshmemx_getmem_on_stream",
        [](intptr_t dst, intptr_t src, size_t elem_size, int pe, intptr_t stream) {
            auto dst_addr = (void *)dst;
            auto src_addr = (void *)src;
            aclrtStream acl_stream = nullptr;
            if (stream != 0) {
                acl_stream = reinterpret_cast<aclrtStream>(stream);
            }
            aclshmemx_getmem_on_stream(dst_addr, src_addr, elem_size, pe, acl_stream);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("elem_size"), py::arg("pe"),
        py::arg("stream"), R"(
    Copy contiguous data from symmetric memory on a remote PE to a local buffer.

    Arguments:
        dst                [in] Pointer on local memory of the destination data.
        src                [in] Pointer on symmetric address of the source data.
        elem_size          [in] Number of elements in the dest and source arrays.
        pe                 [in] PE number of the remote PE.
        stream             [in] copy used stream(use default stream if stream == NULL).
    )");

    m.def(
        "aclshmemx_signal_op_on_stream",
        [](intptr_t sig, int32_t signal, int sig_op, int pe, intptr_t stream) {
            auto sig_addr = (int32_t *)sig;
            aclrtStream acl_stream = nullptr;
            if (stream != 0) {
                acl_stream = reinterpret_cast<aclrtStream>(stream);
            }
            aclshmemx_signal_op_on_stream(sig_addr, signal, sig_op, pe, acl_stream);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("sig_addr"), py::arg("signal"), py::arg("sig_op"),
        py::arg("pe"), py::arg("stream"), R"(
    Performs an atomic operation on a remote signal variable at the specified PE, with the operation executed on the given stream.

    Arguments:
        sig_addr             [in] Local address of the source signal variable that is accessible at the target PE.
        signal               [in] The value to be used in the atomic operation.
        sig_op               [in] The operation to perform on the remote signal. Supported operations:
                                   ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD.
        pe                   [in] The PE number on which the remote signal variable is to be updated.
        stream               [in] Used stream(use default stream if stream == NULL).
        )");

    m.def(
        "aclshmemx_signal_wait_until_on_stream",
        [](intptr_t sig, int cmp, int32_t cmp_val, intptr_t stream) {
            auto sig_addr = (int32_t *)sig;
            aclrtStream acl_stream = nullptr;
            if (stream != 0) {
                acl_stream = reinterpret_cast<aclrtStream>(stream);
            }
            aclshmemx_signal_wait_until_on_stream(sig_addr, cmp, cmp_val, acl_stream);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("sig_addr"), py::arg("cmp"), py::arg("cmp_val"),
        py::arg("stream"), R"(
    Waits until a symmetric signal variable satisfies a given condition. The wait
    is performed on the specified stream; the call returns immediately on the host.
    When the stream is synchronized, the condition is guaranteed to be true.

    Arguments:
        sig_addr             [in] Local address of the source signal variable.
        cmp                  [in] The comparison op that compares sig_addr with cmp_val. Supported operators:
                                  ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/
                                  ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
        cmp_val              [in] The value against which the object pointed to by sig_addr will be compared.
        stream               [in] ACL stream used for execution ordering. Use default stream if stream == NULL.
        )");

#define PYBIND_ACLSHMEM_PUT_SIZE_SIGNAL(BITS)                                                                         \
    {                                                                                                                 \
        std::string funcName = "aclshmemx_put" #BITS "_signal";                                                       \
        m.def(                                                                                                        \
            funcName.c_str(),                                                                                         \
            [](intptr_t dst, intptr_t src, size_t nelems, intptr_t sig, int32_t signal, int sig_op, int pe) {         \
                auto dst_addr = (void *)dst;                                                                          \
                auto src_addr = (void *)src;                                                                          \
                auto sig_addr = (int32_t *)sig;                                                                       \
                aclshmem_put##BITS##_signal(dst_addr, src_addr, nelems, sig_addr, signal, sig_op, pe);                \
            },                                                                                                        \
            py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("nelems"),              \
            py::arg("sig"), py::arg("signal"), py::arg("sig_op"), py::arg("pe"), R"(                                  \
        Synchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and
        updating a remote signal flag on completion.
                                                                                                                      \
        Arguments:                                                                                                    \
            dst               [in] Pointer on Symmetric address of the destination data.
            src               [in] Pointer on local memory of the source data.
            elem_size         [in] Number of elements in the dest and source arrays.
            sig_addr          [in] Symmetric address of the signal word to be updated.
            signal            [in] The value used to update sig_addr.
            sig_op            [in] Operation used to update sig_addr with signal.
                                   Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
            pe                [in] PE number of the remote PE.
            )");                                                                                                      \
    }

    ACLSHMEM_SIZE_FUNC(PYBIND_ACLSHMEM_PUT_SIZE_SIGNAL)
#undef PYBIND_ACLSHMEM_PUT_SIZE_SIGNAL

#define PYBIND_ACLSHMEM_PUT_SIZE_SIGNAL_NBI(BITS)                                                                     \
    {                                                                                                                 \
        std::string funcName = "aclshmemx_put" #BITS "_signal_nbi";                                                   \
        m.def(                                                                                                        \
            funcName.c_str(),                                                                                         \
            [](intptr_t dst, intptr_t src, size_t nelems, intptr_t sig, int32_t signal, int sig_op, int pe) {         \
                auto dst_addr = (void *)dst;                                                                          \
                auto src_addr = (void *)src;                                                                          \
                auto sig_addr = (int32_t *)sig;                                                                       \
                aclshmem_put##BITS##_signal_nbi(dst_addr, src_addr, nelems, sig_addr, signal, sig_op, pe);            \
            },                                                                                                        \
            py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"), py::arg("nelems"),              \
            py::arg("sig"), py::arg("signal"), py::arg("sig_op"), py::arg("pe"), R"(                                  \
        Asynchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and
        updating a remote signal flag on completion.
                                                                                                                      \
        Arguments:                                                                                                    \
            dst               [in] Pointer on symmetric address of the destination data.
            src               [in] Pointer on local memory of the source data.
            elem_size         [in] Number of elements in the dest and source arrays.
            sig_addr          [in] Symmetric address of the signal word to be updated.
            signal            [in] The value used to update sig_addr.
            sig_op            [in] Operation used to update sig_addr with signal.
                                   Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
            pe                [in] PE number of the remote PE.
            )");                                                                                                      \
    }

    ACLSHMEM_SIZE_FUNC(PYBIND_ACLSHMEM_PUT_SIZE_SIGNAL_NBI)
#undef PYBIND_ACLSHMEM_PUT_SIZE_SIGNAL_NBI

    m.def("aclshmem_global_exit", &aclshmem_global_exit, py::call_guard<py::gil_scoped_release>(), py::arg("status"), R"(
Exit all ranks process by broadcast all ranks to exit();

Arguments:
    status(int): the status which is provided to exit();
    )");

    m.def("aclshmemx_quiet_on_stream",
        [](intptr_t stream) {
            aclrtStream acl_stream = nullptr;
            if (stream != 0) {
                acl_stream = reinterpret_cast<aclrtStream>(stream);
            }
            aclshmemx_quiet_on_stream(acl_stream);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("stream"), R"(
Ensures completion of all previously issued operations on symmetric data objects
issued by the calling PE. The quiet is queued on the given stream; the caller
must synchronize the stream to observe completion from the host.

Arguments:
    stream               [in] ACL stream on which to queue the quiet operation.
    )");

    m.def("my_pe", &aclshmem_team_my_pe, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Get my PE number within a team, i.e. index of the PE

Arguments:
    team(int): team id
Returns:
    On success, returns the PE's number in the specified team. On error, -1 is returned.
    )");

    m.def("pe_count", &aclshmem_team_n_pes, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Get number of PEs with in a team, i.e. how many PEs in the team.

Arguments:
    team(int): team id
Returns:
    On success, returns total number of PEs in the specified team. On error, -1 is returned.
    )");

    m.def("set_log_level", &aclshmemx_set_log_level, py::call_guard<py::gil_scoped_release>(), py::arg("level"), R"(
set all module log on level;

Arguments:
    level(int): the status which is provided to exit();
    )");

    m.def("set_extern_logger", &shm::aclshmem_set_extern_logger_py, py::call_guard<py::gil_scoped_release>(),
          py::arg("func"), R"(
set log function of all module;

Arguments:
    func(void (*func)(int level, const char *msg)): function of log;
    )");

m.def("aclshmem_signal_wait_until",
    [](intptr_t sig_addr, int cmp, int cmp_val) {
        aclshmem_signal_wait_until(
            reinterpret_cast<int32_t*>(sig_addr),
            cmp,
            static_cast<int32_t>(cmp_val)
        );
    },
    py::call_guard<py::gil_scoped_release>(),
    py::arg("sig_addr"),
    py::arg("cmp"),
    py::arg("cmp_val"),
    R"(
Wait until the value at signal address satisfies a comparison condition.

This function blocks the calling thread until:
    *sig_addr cmp cmp_val  is true,
where OP is determined by 'cmp' (e.g., CMP_EQ, CMP_NE, etc.).

Arguments:
    sig_addr(int): Local address of the source signal variable.
    cmp(int): Comparison op.
    cmp_val(int): Value to compare against.
Returns:
    The value of sig_addr when condition is met.
    )");

#define PYBIND_ACLSHMEM_WAIT_UNTIL(NAME, TYPE)                                                                         \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_wait_until";                                                        \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivar, int cmp, TYPE cmp_value) {                                                               \
                auto ivar_addr = (TYPE *)ivar;                                                                         \
                aclshmem_##NAME##_wait_until(ivar_addr, cmp, cmp_value);                                               \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(), py::arg("ivar"), py::arg("cmp"), py::arg("cmp_value"),           \
            R"(                                                                                                        \
    Wait until the element at the symmetric address satisfies a comparison condition.                                  \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivar            [in] Symmetric address of a remotely accessible data object.
        cmp             [in] Comparison op.
        cmp_value       [in] Value to compare against.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_WAIT_UNTIL);
#undef PYBIND_ACLSHMEM_WAIT_UNTIL

#define PYBIND_ACLSHMEM_WAIT(NAME, TYPE)                                                                               \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_wait";                                                              \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivar, TYPE cmp_value) {                                                                        \
                auto ivar_addr = (TYPE *)ivar;                                                                         \
                aclshmem_##NAME##_wait(ivar_addr, cmp_value);                                                          \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(), py::arg("ivar"), py::arg("cmp_value"),                           \
            R"(                                                                                                        \
    Wait until the element at the symmetric address is not equal to cmp_value.                                         \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivar            [in] Symmetric address of a remotely accessible data object.
        cmp_value       [in] Value to compare against.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_WAIT);
#undef PYBIND_ACLSHMEM_WAIT

#define PYBIND_ACLSHMEM_WAIT_UNTIL_ALL(NAME, TYPE)                                                                     \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_wait_until_all";                                                    \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t status_ptr, int cmp, TYPE cmp_value) {                      \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                aclshmem_##NAME##_wait_until_all(ivar_addrs, nelems, status, cmp, cmp_value);                          \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_value"),                                                                                      \
            R"(                                                                                                        \
    Wait until all elements in the symmetric array satisfy the condition.                                              \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_value       [in] Value to compare against.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_WAIT_UNTIL_ALL);
#undef PYBIND_ACLSHMEM_WAIT_UNTIL_ALL

#define PYBIND_ACLSHMEM_WAIT_UNTIL_ANY(NAME, TYPE)                                                                     \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_wait_until_any";                                                    \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t status_ptr, int cmp, TYPE cmp_value, intptr_t res_out_ptr) {\
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto res_out = reinterpret_cast<size_t *>(res_out_ptr);                                                \
                aclshmem_##NAME##_wait_until_any(ivar_addrs, nelems, status, cmp, cmp_value, res_out);                 \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_value"),                                                                                      \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Wait until at least one element in the symmetric array satisfy the condition.                                      \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_value       [in] Value to compare against.
        res_out         [out] Return value.
    Returns:
        The index of an element in the ivars array that satisfies the wait condition.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_WAIT_UNTIL_ANY);
#undef PYBIND_ACLSHMEM_WAIT_UNTIL_ANY

#define PYBIND_ACLSHMEM_WAIT_UNTIL_SOME(NAME, TYPE)                                                                    \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_wait_until_some";                                                   \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t indices_ptr, intptr_t status_ptr, int cmp, TYPE cmp_value,  \
                intptr_t res_out_ptr) {                                                                                \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto indices = reinterpret_cast<size_t *>(indices_ptr);                                                \
                auto res_out = reinterpret_cast<size_t *>(res_out_ptr);                                                \
                aclshmem_##NAME##_wait_until_some(ivar_addrs, nelems, indices, status, cmp, cmp_value, res_out);       \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("indices"),                                                                                        \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_value"),                                                                                      \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Wait until at least one element in the symmetric array satisfy the condition.                                      \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        indices         [out] Local address of an array of indices of length at least nelems into ivars that
                             satisfied the wait condition.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_value       [in] Value to compare against.
        res_out         [out] Return value.
    Returns:
        The number of indices returned in the indices array.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_WAIT_UNTIL_SOME);
#undef PYBIND_ACLSHMEM_WAIT_UNTIL_SOME

#define PYBIND_ACLSHMEM_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)                                                              \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_wait_until_all_vector";                                             \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t status_ptr, int cmp, intptr_t cmp_values_ptr) {             \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto cmp_values = (TYPE *)cmp_values_ptr;                                                              \
                aclshmem_##NAME##_wait_until_all_vector(ivar_addrs, nelems, status, cmp, cmp_values);                  \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_values"),                                                                                     \
            R"(                                                                                                        \
    Wait until all elements in the symmetric array satisfy the condition.                                              \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_values       [in] Value to compare against which is an array.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_WAIT_UNTIL_ALL_VECTOR);
#undef PYBIND_ACLSHMEM_WAIT_UNTIL_ALL_VECTOR

#define PYBIND_ACLSHMEM_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                                                              \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_wait_until_any_vector";                                             \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t status_ptr, int cmp, intptr_t cmp_values_ptr,               \
                intptr_t res_out_ptr) {                                                                                \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto cmp_values = (TYPE *)cmp_values_ptr;                                                              \
                auto res_out = reinterpret_cast<size_t *>(res_out_ptr);                                                \
                aclshmem_##NAME##_wait_until_any_vector(ivar_addrs, nelems, status, cmp, cmp_values, res_out);         \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_values"),                                                                                     \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Wait until at least one element in the symmetric array satisfy the condition.                                      \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_values      [in] Value to compare against which is an array.
        res_out         [out] Return value.
    Returns:
        The index of an element in the ivars array that satisfies the wait condition.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_WAIT_UNTIL_ANY_VECTOR);
#undef PYBIND_ACLSHMEM_WAIT_UNTIL_ANY_VECTOR

#define PYBIND_ACLSHMEM_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                                             \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_wait_until_some_vector";                                            \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t indices_ptr, intptr_t status_ptr, int cmp,                  \
                intptr_t cmp_values_ptr, intptr_t res_out_ptr) {                                                       \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto cmp_values = (TYPE *)cmp_values_ptr;                                                              \
                auto indices = reinterpret_cast<size_t *>(indices_ptr);                                                \
                auto res_out = reinterpret_cast<size_t *>(res_out_ptr);                                                \
                aclshmem_##NAME##_wait_until_some_vector(                                                              \
                    ivar_addrs, nelems, indices, status, cmp, cmp_values, res_out                                      \
                );                                                                                                     \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("indices"),                                                                                        \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_values"),                                                                                     \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Wait until at least one element in the symmetric array satisfy the condition.                                      \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        indices         [out] Local address of an array of indices of length at least nelems into ivars that
                             satisfied the wait condition.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_values      [in] Value to compare against which is an array.
        res_out         [out] Return value.
    Returns:
        The number of indices returned in the indices array.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_WAIT_UNTIL_SOME_VECTOR);
#undef PYBIND_ACLSHMEM_WAIT_UNTIL_SOME_VECTOR

#define PYBIND_ACLSHMEM_TEST(NAME, TYPE)                                                                               \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_test";                                                              \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivar, int cmp, TYPE cmp_value, intptr_t res_out_ptr) {                                         \
                auto ivar_addr = (TYPE *)ivar;                                                                         \
                auto res_out = reinterpret_cast<int *>(res_out_ptr);                                                   \
                aclshmem_##NAME##_test(ivar_addr, cmp, cmp_value, res_out);                                            \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivar"),                                                                                           \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_value"),                                                                                      \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Check whether the element at the symmetric address satisfies a comparison condition.                               \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivar            [in] Symmetric address of a remotely accessible data object.
        cmp             [in] Comparison op.
        cmp_value       [in] Value to compare against.
        res_out         [out] Return value.
    Returns:
        Returns 1 when the comparison of the symmetric object pointed to by ivar with the value cmp_value according to
        the comparison op cmp; otherwise, it returns 0.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_TEST);
#undef PYBIND_ACLSHMEM_TEST

#define PYBIND_ACLSHMEM_TEST_ANY(NAME, TYPE)                                                                           \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_test_any";                                                          \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t status_ptr, int cmp, TYPE cmp_value, intptr_t res_out_ptr) {\
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto res_out = reinterpret_cast<size_t *>(res_out_ptr);                                                \
                aclshmem_##NAME##_test_any(ivar_addrs, nelems, status, cmp, cmp_value, res_out);                       \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_value"),                                                                                      \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Check whether at least one element in the symmetric array satisfy the condition.                                   \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_value       [in] Value to compare against.
        res_out         [out] Return value.
    Returns:
        Returns the index of an element in the ivars array that satisfies the test condition.
        If the test set is empty or no conditions in the test set are satisfied, it returns SIZE_MAX.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_TEST_ANY);
#undef PYBIND_ACLSHMEM_TEST_ANY

#define PYBIND_ACLSHMEM_TEST_SOME(NAME, TYPE)                                                                          \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_test_some";                                                         \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t indices_ptr, intptr_t status_ptr, int cmp, TYPE cmp_value,  \
                intptr_t res_out_ptr) {                                                                                \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto indices = reinterpret_cast<size_t *>(indices_ptr);                                                \
                auto res_out = reinterpret_cast<size_t *>(res_out_ptr);                                                \
                aclshmem_##NAME##_test_some(ivar_addrs, nelems, indices, status, cmp, cmp_value, res_out);             \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("indices"),                                                                                        \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_value"),                                                                                      \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Check whether at least one element in the symmetric array satisfy the condition.                                   \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        indices         [out] Local address of an array of indices of length at least nelems into ivars that
                             satisfied the wait condition.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_value       [in] Value to compare against.
        res_out         [out] Return value.
    Returns:
        Returns the number of indices returned in the indices array. If the test set is empty, it returns 0.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_TEST_SOME);
#undef PYBIND_ACLSHMEM_TEST_SOME

#define PYBIND_ACLSHMEM_TEST_ALL_VECTOR(NAME, TYPE)                                                                    \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_test_all_vector";                                                   \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t status_ptr, int cmp, intptr_t cmp_values_ptr,               \
                intptr_t res_out_ptr) {                                                                                \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto cmp_values = (TYPE *)cmp_values_ptr;                                                              \
                auto res_out = reinterpret_cast<int *>(res_out_ptr);                                                   \
                aclshmem_##NAME##_test_all_vector(ivar_addrs, nelems, status, cmp, cmp_values, res_out);               \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_values"),                                                                                     \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Check whether all elements in the symmetric array satisfy the condition.                                           \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_values      [in] Value to compare against which is an array.
        res_out         [out] Return value.
    Returns:
        Returns 1 when all variables in ivars satisfy the test conditions or nelems is 0, otherwise it returns 0.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_TEST_ALL_VECTOR);
#undef PYBIND_ACLSHMEM_TEST_ALL_VECTOR

#define PYBIND_ACLSHMEM_TEST_ANY_VECTOR(NAME, TYPE)                                                                    \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_test_any_vector";                                                   \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t status_ptr, int cmp, intptr_t cmp_values_ptr,               \
                intptr_t res_out_ptr) {                                                                                \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto cmp_values = (TYPE *)cmp_values_ptr;                                                              \
                auto res_out = reinterpret_cast<size_t *>(res_out_ptr);                                                \
                aclshmem_##NAME##_test_any_vector(ivar_addrs, nelems, status, cmp, cmp_values, res_out);               \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_values"),                                                                                     \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Check whether at least one element in the symmetric array satisfy the condition.                                   \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_values      [in] Value to compare against which is an array.
        res_out         [out] Return value.
    Returns:
        Returns the index of an element in the ivars array that satisfies the test condition.
        If the test set is empty or no conditions in the test set are satisfied, it returns SIZE_MAX.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_TEST_ANY_VECTOR);
#undef PYBIND_ACLSHMEM_TEST_ANY_VECTOR

#define PYBIND_ACLSHMEM_TEST_SOME_VECTOR(NAME, TYPE)                                                                   \
    {                                                                                                                  \
        std::string funcName = "aclshmem_" #NAME "_test_some_vector";                                                  \
        m.def(                                                                                                         \
            funcName.c_str(),                                                                                          \
            [](intptr_t ivars_ptr, size_t nelems, intptr_t indices_ptr, intptr_t status_ptr, int cmp,                  \
                intptr_t cmp_values_ptr, intptr_t res_out_ptr) {                                                       \
                auto ivar_addrs = (TYPE *)ivars_ptr;                                                                   \
                auto status = (status_ptr == 0) ? nullptr : reinterpret_cast<const int *>(status_ptr);                 \
                auto cmp_values = (TYPE *)cmp_values_ptr;                                                              \
                auto indices = reinterpret_cast<size_t *>(indices_ptr);                                                \
                auto res_out = reinterpret_cast<size_t *>(res_out_ptr);                                                \
                aclshmem_##NAME##_test_some_vector(ivar_addrs, nelems, indices, status, cmp, cmp_values, res_out);     \
            },                                                                                                         \
            py::call_guard<py::gil_scoped_release>(),                                                                  \
            py::arg("ivars"),                                                                                          \
            py::arg("nelems"),                                                                                         \
            py::arg("indices"),                                                                                        \
            py::arg("status"),                                                                                         \
            py::arg("cmp"),                                                                                            \
            py::arg("cmp_values"),                                                                                     \
            py::arg("res_out"),                                                                                        \
            R"(                                                                                                        \
    Check whether at least one element in the symmetric array satisfy the condition.                                   \
                                                                                                                       \
    Arguments:                                                                                                         \
        ivars           [in] Symmetric address of an array of remotely accessible data objects.
        nelems          [in] The number of elements in the ivars array.
        indices         [out] Local address of an array of indices of length at least nelems into ivars that
                             satisfied the wait condition.
        status          [in] Local address of an optional mask array of length nelems that indicates which
                             elements in ivars are excluded from the wait set.
        cmp             [in] Comparison op.
        cmp_values      [in] Value to compare against which is an array.
        res_out         [out] Return value.
    Returns:
        Returns the number of indices returned in the indices array. If the test set is empty, it returns 0.
        )");                                                                                                           \
    }

    ACLSHMEM_P2P_SYNC_TYPE_FUNC(PYBIND_ACLSHMEM_TEST_SOME_VECTOR);
#undef PYBIND_ACLSHMEM_TEST_SOME_VECTOR

}
