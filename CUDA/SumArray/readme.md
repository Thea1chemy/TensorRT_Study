在GPU上实现矩阵相加（C = A + B）
nvcc sum_arrays.cu -o sum
nsys profile -o sum -f ./sum
然后用nsystem打开sum.qdrep观察CUDA API那一栏的调用情况
