import sys
import random

ni = int(sys.argv[1])
nj = int(sys.argv[2])
nk = int(sys.argv[3])


#  ifdef MINI_DATASET
#   define NI 20
#   define NJ 25
#   define NK 30
#  endif

#  ifdef SMALL_DATASET
#   define NI 60
#   define NJ 70
#   define NK 80
#  endif


ni*nk*nj

def gemm(X,Y,Z, alpha, beta):
    # General matrix multiplication (Z = a*XY + b*Z)
    for i in range(ni):
        for j in range(nj):
            sum = 0
            for l in range(nk):
                sum += X[i * nk + l] * Y[l * nj + j]
            Z[i * nj + j] = int(alpha * sum + beta * Z[i * nj + j])


alpha = 32412
beta  = 2123
X = [int(random.random() * 100 - 50) for _ in range(ni*nk)]
Y = [int(random.random() * 100 - 50) for _ in range(nk*nj)]
Z = [int(random.random() * 100 - 50) for _ in range(ni*nj)]

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
print("#define NI {}".format(ni))
print("#define NJ {}".format(nj))
print("#define NK {}".format(nk))

print_array_interleaved("int32_t", "inputX", "NI*NK", X)
print_array_interleaved("int32_t", "inputY", "NK*NJ", Y)
print_array_interleaved("int32_t", "inputZ", "NI*NJ", Z)
gemm(X,Y,Z, alpha, beta )
print_array("int32_t", "expected_result", "NI*NJ", Z)
