# Vendored offline-build dependencies (Ascend / Atlas A3)

Regenerate with `bash scripts/vendor_deps.sh` on a networked Linux machine,
then commit. Consumed by `scripts/build_ascend_a3.sh` with **no network
access** at build time. Each tree carries a `.vendor-sha` provenance file.

| Path | Upstream | Commit / Version | State |
|---|---|---|
| `llvm-project/` | github.com/llvm/llvm-project | `fad3272286528b8a491085183434c5ad4b59ab92` | triton-ascend `llvm_patch/fad3272.patch` **pre-applied**; trimmed to `llvm/+mlir/+lld`, test suites removed (build with `-DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF`; FileCheck is kept, llvm-lit is not built and not needed) |
| `triton-ascend/` | github.com/triton-lang/triton-ascend | `bfd8f55d124f0da70ef792cb8d9f759d7302bb26` | pristine (submodule pin; `3rdparty/triton-ascend.patch` applies -- verified by vendor_deps.sh and build-step preflight; setup.py applies it at build time) |
| `triton-ascend/third_party/ascend/AscendNPU-IR/` | gitcode.com/Ascend/AscendNPU-IR | `1b33649151424798b95187fb5fc4824f982607e4` | pristine (triton-ascend's pin; setup.py applies `3rdparty/AscendNPU-IR.patch`; TA cmake builds it with `BISHENGIR_BUILD_STANDALONE_IR_ONLY=ON`, its `third-party/` is not needed) |
| `AscendNPU-IR/` | gitcode.com/Ascend/AscendNPU-IR | `1b33649151424798b95187fb5fc4824f982607e4` | standalone bisheng tool build (build-step 3); its `third-party/llvm-project` @ `cd708029e0b2869e80abe31ddb175f7c35361f90` has npuir's own `build-tools/patches/llvm-project/*.patch` **pre-applied** and is trimmed like above; `third-party/torch-mlir` intentionally absent (`BUILD_TORCH_MLIR=OFF`) |
| `shmem/` | gitcode.com/cann/shmem | `81c95bad8b943fbcf99e1a9664350aa41a937588` | pristine (submodule pin; the `-python_extension` path of its `scripts/build.sh` performs no downloads -- catlass/googletest/json are only fetched by -uttests/-examples/-python_example/-full/SOC_TYPE=Ascend950) |
| `nlohmann-json/` | github.com/nlohmann/json release | v3.11.3 | `include/` + `single_include/`; passed to setup.py via `JSON_SYSPATH` under `TRITON_OFFLINE_BUILD=1` |

Notes:

* `3rdparty/triton-ascend` and `3rdparty/shmem` are **no longer git
  submodules** (gitlinks and `.gitmodules` entries removed by vendor_deps.sh).
  `3rdparty/triton` and `3rdparty/mori` remain submodules for the
  NVIDIA/ROCm flows.
* `3rdparty/.gitattributes` forces LF everywhere so a Windows checkout can
  never corrupt the vendored trees or `3rdparty/*.patch`.
* The two LLVM trees are trimmed: clang/lldb/flang/libcxx/... and all
  `test/`+`unittests/` directories are absent. This is safe because the
  builds only enable `mlir;llvm;lld` (resp. `mlir`) and pass
  `*_INCLUDE_TESTS=OFF`; FileCheck lives under `llvm/utils` (gated by
  `LLVM_INCLUDE_UTILS`, default ON) and is still built+installed. The
  `mlir-doc` target that setup.py builds comes from the *installed*
  `MLIRConfig.cmake` (an empty aggregator target), not from the source tree.
* Vendored on: 2026-09-12T12:51:32Z by scripts/vendor_deps.sh
