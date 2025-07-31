# /bin/bash!
make app PROJECT=$1 COMPILER_FLAGS=-O3 COMPILER_PREFIX=riscv32-corev- ARCH=rv32imc_zicsr_zifencei_xcvhwlp0p1_xcvmem0p1_xcvmac0p1_xcvbi0p1_xcvalu0p1_xcvsimd0p1_xcvbitmanip0p1
