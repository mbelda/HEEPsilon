# /bin/bash!
make app PROJECT=$1
cd build/eslepfl_systems_cgra-x-heep_0/sim-verilator
./Vtestharness +firmware=../../../sw/build/main.hex
cd ../../..