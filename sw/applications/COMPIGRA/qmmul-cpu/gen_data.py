import sys
import random

# Read the base size from arguments
size = int(sys.argv[1])

# For a square-like setup based on 'size'
NI = size
NJ = size
NK = size

# 1-D Array sizes
A_size = NI * NK
B_size = NK * NJ
bias_size = NI
scale_size = NJ
C_size = NI * NJ

# Generate random 1-D data
A = [int(random.random() * 100.0 - 50.0) for _ in range(A_size)]
B = [int(random.random() * 100.0 - 50.0) for _ in range(B_size)]
bias = [int(random.random() * 20.0 - 10.0) for _ in range(bias_size)]
scale = [int(random.random() * 5.0 + 1.0) for _ in range(scale_size)]

# Python implementation of your qmmul to generate expected data (using 1D indexing)
expected_C = [0] * C_size
for i in range(NI):
    for j in range(NJ):
        acc = 0
        for k in range(NK):
            # 2D to 1D mapping: [row * total_columns + col]
            a = A[i * NK + k] + bias[i]
            b = B[k * NJ + j] * scale[j]
            acc += a * b
        expected_C[i * NJ + j] = acc

# Formatting helper functions
def print_array_interleaved(array_type, array_name, array_sz, pyarr):
    print("volatile {} {}[{}] __attribute__((section(\".xheep_data_interleaved\"))) = ".format(array_type, array_name, array_sz))
    print("{")
    print(", ".join(map(str, pyarr)))
    print("};")

def print_array(array_type, array_name, array_sz, pyarr):
    print("volatile {} {}[{}] = ".format(array_type, array_name, array_sz))
    print("{")
    print(", ".join(map(str, pyarr)))
    print("};")

# Print C Header Outputs
print("#include <stdint.h>")
print("#define NI {}".format(NI))
print("#define NJ {}".format(NJ))
print("#define NK {}".format(NK))
print()

# Print input matrices, biases, scales, and expected output as 1-D C arrays
print_array_interleaved("int32_t", "matrix_A", NI * NK, A)
print_array_interleaved("int32_t", "matrix_B", NK * NJ, B)
print_array("int32_t", "bias", NI, bias)
print_array("int32_t", "scale", NJ, scale)
print_array("int32_t", "expected_result", NI * NJ, expected_C)