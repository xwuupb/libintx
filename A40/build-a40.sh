#!/usr/bin/env bash
#
# XW: please make sure that you have the software ready
module reset
module load compiler/GCC/11.3.0 system/CUDA/11.8.0 devel/CMake/3.23.1-GCCcore-11.3.0
#
# XW: please
# - use appropriate path
# - use -DCMAKE_CUDA_ARCHITECTURES=70 for NVIDIA V100
cmake -DCMAKE_BUILD_TYPE=Release -S /dev/shm/libintx -B /dev/shm/gpucuda -DLIBINTX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLIBINTX_MAX_L=3
cmake --build /dev/shm/gpucuda --target libintx.cuda.benchmarks.fpl24 -j $(nproc)
