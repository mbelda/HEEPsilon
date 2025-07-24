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
print("#define ROWS_A {}".format(ni))
print("#define COLS_B {}".format(nj))
print("#define COLS_A {}".format(nk))
print("#define COLS_D {}".format(nl))
print("#define COLS_C {}".format(nm))

print("#define ROWS_C COLS_B")
print("#define ROWS_E ROWS_A")
print("#define COLS_E COLS_B")
print("#define ROWS_F COLS_B")
print("#define COLS_F COLS_D")


print_array_interleaved("int32_t", "matrixA", "ROWS_A*COLS_A", A)
print_array_interleaved("int32_t", "matrixB", "COLS_A*COLS_B", B)
print_array_interleaved("int32_t", "matrixC", "COLS_B*COLS_C", C)
print_array_interleaved("int32_t", "matrixD", "COLS_C*COLS_D", D)
print_array_interleaved("int32_t", "matrixE", "ROWS_A*COLS_B", [0])
print_array_interleaved("int32_t", "matrixF", "COLS_B*COLS_D", [0])
print_array_interleaved("int32_t", "matrixG", "ROWS_A*COLS_D", [0])
print_array("int32_t", "expected_result", "ROWS_A*COLS_D", G)
