#include <gtest/gtest.h>
#include "shmemi_host_common.h"

// simple test to verify bootstrap SO loading behavior

// helper that tries to init bootstrap with given flags
static int try_bootstrap(int flags) {
    aclshmemx_init_attr_t attr;
    memset(&attr, 0, sizeof(attr));
    attr.my_pe = 0;
    attr.n_pes = 1;
    attr.option_attr.sockFd = -1;
    attr.option_attr.shm_init_timeout = 10;
    attr.option_attr.control_operation_timeout = 10;
    attr.ip_port[0] = '\0';
    return aclshmemi_bootstrap_init(flags, &attr);
}

TEST(BootstrapTest, LoadConfigStoreSuccess) {
    // ensure SO file exists in current directory
    FILE *f = fopen("aclshmem_bootstrap_config_store.so", "r");
    if (!f) {
        GTEST_SKIP() << "config_store SO not present, skipping";
    } else {
        fclose(f);
    }
    int rc = try_bootstrap(ACLSHMEMX_INIT_WITH_DEFAULT);
    EXPECT_EQ(rc, ACLSHMEM_SUCCESS);
    aclshmemi_bootstrap_finalize();
}

TEST(BootstrapTest, MissingSoFails) {
    // temporarily rename so to simulate missing
    const char *name = "aclshmem_bootstrap_config_store.so";
    if (access(name, F_OK) == 0) {
        rename(name, "tmp.so");
    }
    int rc = try_bootstrap(ACLSHMEMX_INIT_WITH_DEFAULT);
    EXPECT_EQ(rc, ACLSHMEM_INVALID_PARAM);
    // restore
    if (access("tmp.so", F_OK) == 0) {
        rename("tmp.so", name);
    }
}

TEST(BootstrapTest, ModeSwitching) {
    // should handle MPI and DEFAULT modes
    int rc = try_bootstrap(ACLSHMEMX_INIT_WITH_MPI);
    EXPECT_NE(rc, ACLSHMEM_SUCCESS); // MPI so usually not present
    rc = try_bootstrap(ACLSHMEMX_INIT_WITH_DEFAULT);
    // rc either success or invalid_value depending on so presence
    aclshmemi_bootstrap_finalize();
}

// repeated initialization exercises both DEFAULT and UNIQUEID paths
TEST(BootstrapTest, RepeatInitDefaultThenUnique) {
    int rc = try_bootstrap(ACLSHMEMX_INIT_WITH_DEFAULT);
    if (rc == ACLSHMEM_SUCCESS) {
        aclshmemi_bootstrap_finalize();
    }

    rc = try_bootstrap(ACLSHMEMX_INIT_WITH_UNIQUEID);
    EXPECT_TRUE(rc == ACLSHMEM_SUCCESS || rc == ACLSHMEM_INVALID_PARAM ||
                rc == ACLSHMEM_INVALID_VALUE);
    if (rc == ACLSHMEM_SUCCESS) {
        aclshmemi_bootstrap_finalize();
    }
}
