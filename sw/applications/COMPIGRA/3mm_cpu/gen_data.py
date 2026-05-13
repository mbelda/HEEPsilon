import sys
import random

ni = int(sys.argv[1])
nj = int(sys.argv[2])
nk = int(sys.argv[3])
nl = int(sys.argv[4])
nm = int(sys.argv[5])

#  ifdef MINI_DATASET
#   define NI 16
#   define NJ 18
#   define NK 20
#   define NL 22
#   define NM 24
#  endif

#  ifdef SMALL_DATASET
#   define NI 40
#   define NJ 50
#   define NK 60
#   define NL 70
#   define NM 80
#  endif


def mmul(X,Y,Z, rows, cols, intermediate_sz):
    # matrix multiplication (Z = A*B)
    for i in range(rows):
        for j in range(cols):
            sum = 0
            for l in range(intermediate_sz):
                sum += X[i * intermediate_sz + l] * Y[l * cols + j]
            Z[i * cols + j] = sum


A = [int(random.random() * 100 - 50) for _ in range(ni*nk)]
B = [int(random.random() * 100 - 50) for _ in range(nk*nj)]
E = [0 for _ in range(ni*nj)]
mmul(A,B,E, ni, nj, nk)
C = [int(random.random() * 100 - 50) for _ in range(nj*nm)]
D = [int(random.random() * 100 - 50) for _ in range(nm*nl)]
F = [0 for _ in range(nj*nl)]
mmul(C,D,F, nj, nl, nm)
G = [0 for _ in range(ni*nl)]
mmul(E,F,G, ni, nl, nj)


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
print("#define NI {}".format(ni))
print("#define NJ {}".format(nj))
print("#define NK {}".format(nk))
print("#define NL {}".format(nl))
print("#define NM {}".format(nm))

print_array_interleaved("int32_t", "inputA", "NI*NK", A)
print_array_interleaved("int32_t", "inputB", "NK*NJ", B)
print_array_interleaved("int32_t", "inputC", "NJ*NM", C)
print_array_interleaved("int32_t", "inputD", "NM*NL", D)
print_array_interleaved("int32_t", "outputE", "NI*NJ", [0])
print_array_interleaved("int32_t", "outputF", "NJ*NL", [0])
print_array_interleaved("int32_t", "outputG", "NI*NL", [0])
print_array("int32_t", "expected_result", "NI*NL", G)
