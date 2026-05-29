import sys
import random

rows_A = int(sys.argv[1])
cols_A = int(sys.argv[2])
cols_B = int(sys.argv[3])


A = [int(random.random() * 100 - 50) for _ in range(rows_A*cols_A)]
B = [int(random.random() * 100 - 50) for _ in range(cols_A*cols_B)]

def print_array_interleaved_volatile(array_type, array_name, array_sz, pyarr):
    print("volatile {} {}[{}] __attribute__((section(\".xheep_data_interleaved\"))) = ".format(array_type, array_name, array_sz))
    print("{")
    print(", ".join(map(str, pyarr)))
    print("};")

def print_array_interleaved(array_type, array_name, array_sz, pyarr):
    print("{} {}[{}] __attribute__((section(\".xheep_data_interleaved\"))) = ".format(array_type, array_name, array_sz))
    print("{")
    print(", ".join(map(str, pyarr)))
    print("};")

def print_array_volatile(array_type, array_name, array_sz, pyarr):
    print("volatile {} {}[{}] = ".format(array_type, array_name, array_sz))
    print("{")
    print(", ".join(map(str, pyarr)))
    print("};")

def print_array(array_type, array_name, array_sz, pyarr):
    print("{} {}[{}] = ".format(array_type, array_name, array_sz))
    print("{")
    print(", ".join(map(str, pyarr)))
    print("};")

def print_scalar(scalar_type, scalar_name, pyscalar):
    print("{} {} = {};".format(scalar_type, scalar_name, pyscalar))

print("#include <stdint.h>")
print("#define ROWS_A {}".format(rows_A))
print("#define COLS_A {}".format(cols_A))
print("#define COLS_B {}".format(cols_B))
print("#define ROWS_B COLS_A")
print("#define ROWS_C ROWS_A")
print("#define COLS_C COLS_B")

print_array_interleaved("int32_t", "matrixA", "ROWS_A*COLS_A", A)
print_array_interleaved("int32_t", "matrixB", "COLS_A*COLS_B", B)
print_array_interleaved_volatile("int32_t", "matrixC", "ROWS_A*COLS_B", [0 for _ in range(rows_A*cols_B)])
print_array("int32_t", "cpu_out", "ROWS_A*COLS_B", [0 for _ in range(rows_A*cols_B)])
