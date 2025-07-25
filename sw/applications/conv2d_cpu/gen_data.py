import sys
import random

n = int(sys.argv[1])
m = int(sys.argv[2])

def conv2d_cpu(input, output, height, width, filter):
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            acc = 0
            for ki in range(3):
                for kj in range(3):
                    img_i = i + ki - 1
                    img_j = j + kj - 1
                    img_idx = img_i * width + img_j
                    ker_idx = ki * 3 + kj
                    acc += input[img_idx] * filter[ker_idx]
            output[i * width + j] = acc



Image = [int(random.random() * 100 - 50) for _ in range(n*m)]
Filter = [int(random.random() * 10 - 5) for _ in range(3*3)]
Output = [0 for _ in range(n*m)]
conv2d_cpu(Image, Output, n, m, Filter)


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
print("#define IM_HEIGHT {}".format(n))
print("#define IM_WIDTH {}".format(m))

print_array_interleaved("int32_t", "image", "IM_HEIGHT*IM_WIDTH", Image)
print_array_interleaved("int32_t", "output", "IM_HEIGHT*IM_WIDTH", [0])
print_array("int32_t", "filter", "9", Filter)
print_array("int32_t", "expected_output", "(IM_HEIGHT -1)*(IM_WIDTH -1)", Output)
