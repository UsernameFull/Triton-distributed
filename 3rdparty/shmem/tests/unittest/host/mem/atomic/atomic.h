#ifndef ATOMIC_H
#define ATOMIC_H

#include <string>
#include "acl/acl.h"

inline bool is_hardware_atomic_supported()
{
    const char* soc_name = aclrtGetSocName();
    if (soc_name != nullptr && std::string(soc_name).find("Ascend950") != std::string::npos) {
        return true;
    }
    return false;
}

#endif // ATOMIC_H
