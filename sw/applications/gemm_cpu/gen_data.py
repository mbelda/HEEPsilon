import sys
import random

n = int(sys.argv[1])
m = int(sys.argv[2])
k = int(sys.argv[3])

#  ifdef MINI_DATASET
#   define NI 32
#   define NJ 32
#   define NK 32
#  endif

#  ifdef SMALL_DATASET
#   define NI 128
#   define NJ 128
#   define NK 128
#  endif




def gemm(X,Y,Z, alpha, beta):
    # General matrix multiplication (Z = a*XY + b*Z)
    for i in range(n):
        for j in range(k):
            sum = 0
            for l in range(m):
                sum += X[i * m + l] * Y[l * k + j]
            Z[i * k + j] = int(alpha * sum + beta * Z[i * k + j])


alpha = 32412
beta  = 2123
X = [int(random.random() * 100 - 50) for _ in range(n*m)]
Y = [int(random.random() * 100 - 50) for _ in range(m*k)]
Z = [int(random.random() * 100 - 50) for _ in range(n*k)]

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

def print_scalar(scalar_type, scalar_name, pyscalar):
    print("{} {} = {};".format(scalar_type, scalar_name, pyscalar))

print("#include <stdint.h>")
print("#define ALPHA {}".format(alpha))
print("#define BETA {}".format(beta))
print("#define DIM_N {}".format(n))
print("#define DIM_M {}".format(m))
print("#define DIM_K {}".format(k))

print_array("int32_t", "inputX", "DIM_N*DIM_M", X)
print_array("int32_t", "inputY", "DIM_M*DIM_K", Y)
print_array("int32_t", "inputZ", "DIM_N*DIM_K", Z)
gemm(X,Y,Z, alpha, beta )
print_array("int32_t", "expected_result", "DIM_N*DIM_K", Z)
