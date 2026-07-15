import sys
import random

im_height = int(sys.argv[1])
im_width = int(sys.argv[2])

def conv2d_cpu(image_data, filter_data, rowsIm, colsIm):
    output = [0] * (rowsIm * colsIm)
    for i in range(1, rowsIm - 1):
        for j in range(1, colsIm - 1):
            acc = 0
            for fi in range(3):
                for fj in range(3):
                    im_i = i + fi - 1
                    im_j = j + fj - 1
                    im_index = im_i * colsIm + im_j
                    filt_index = fi * 3 + fj
                    acc += image_data[im_index] * filter_data[filt_index]
            output[i * colsIm + j] = acc
    return output

image = [int(random.random() * 100 - 50) for _ in range(im_height * im_width)]
filter = [int(random.random() * 10 - 5) for _ in range(9)]
output = [0 for _ in range(im_height * im_width)]
expected_output = conv2d_cpu(image, filter, im_height, im_width)

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
print("#define IM_HEIGHT {}".format(im_height))
print("#define IM_WIDTH {}".format(im_width))

print_array_interleaved("int32_t", "image", "IM_HEIGHT*IM_WIDTH", image)
print_array("int32_t", "filter", "9", filter)
print_array_interleaved("int32_t", "output", "IM_HEIGHT*IM_WIDTH", output)
print_array("int32_t", "expected_result", "IM_HEIGHT*IM_WIDTH", expected_output)
