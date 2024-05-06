#!/usr/bin/env bash
#
# XW: please make sure that you have the software ready
module reset
module load compiler/GCC/11.3.0 system/CUDA/11.8.0 devel/CMake/3.23.1-GCCcore-11.3.0
#
{
for i in \
    ssss \
    psss \
    psps \
    ppss \
    ppps \
    pppp \
    dsss \
    dsps \
    dspp \
    dsds \
    dpss \
    dpps \
    dppp \
    dpds \
    dpdp \
    ddss \
    ddps \
    ddpp \
    ddds \
    dddp \
    dddd \
    fsss \
    fsps \
    fspp \
    fsds \
    fsdp \
    fsdd \
    fsfs \
    fpss \
    fpps \
    fppp \
    fpds \
    fpdp \
    fpdd \
    fpfs \
    fpfp \
    fdss \
    fdps \
    fdpp \
    fdds \
    fddp \
    fddd \
    fdfs \
    fdfp \
    fdfd \
    ffss \
    ffps \
    ffpp \
    ffds \
    ffdp \
    ffdd \
    fffs \
    fffp \
    fffd \
    ffff;\
do
  # XW: please use appropriate path
  ./gpucuda/tests/libintx.cuda.benchmarks.fpl24 ${i}
  # XW: for the 1st GPU with interval of 2 seconds
done & nvidia-smi -i 0 --query-gpu=index,power.draw,memory.used -l 2 --format=csv,noheader; } \
  # XW: XW_measurement.logFile is easy to parse
  | tee XW_measurement.logFile
