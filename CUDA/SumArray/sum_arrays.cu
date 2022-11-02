#include <stdio.h>
#include <assert.h>
#include <sys/time.h>


double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

cudaError_t checkcuda(cudaError_t result)
{
    if(result != cudaSuccess)
    {
        fprintf(stderr, "Cuda runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void SumArrays(float *a, float *b, float *res, const int size)
{
    for(int i = 0; i < (size * size); i++)
    {
        res[i] = a[i] + b[i];
    }

}

__global__ void sumArraysGPU(float *a, float *b, float *res, const int size)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx < (size * size))
    {
        res[thread_idx] = a[thread_idx] + b[thread_idx];
    }
}

bool check_res(float *a, float *b, const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[])
{
    int nElem = (1 << 12); //矩阵的大小
    printf("Vector.size: %d\n", nElem * nElem);

    size_t nByte = nElem * nElem * sizeof(float);
    
    // 在cpu上开辟内存空间
    float *a_h = (float*)malloc(nByte);
    float *b_h = (float*)malloc(nByte);
    float *res_h = (float*)malloc(nByte);
    float *res_from_gpu_h = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    // 在gpu上开辟存储空间
    float *a_d, *b_d, *res_d;
    checkcuda(cudaMalloc((float **)&a_d, nByte));
    checkcuda(cudaMalloc((float **)&b_d, nByte));
    checkcuda(cudaMalloc((float **)&res_d, nByte));

    // init a_h and b_h
    for(int y = 0; y < nElem; y++)
    {
        for(int x = 0; x < nElem; x++)
        {
            a_h[y * nElem + x] = y;
            b_h[y * nElem + x] = x;
        }
    }

    checkcuda(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    checkcuda(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((nElem * nElem + block.x - 1) / block.x);
    // gpu
    double iStart, iElaps;
    iStart = cpuSecond();
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nElem);
    cudaDeviceSynchronize(); // 这里如果不加同步，调用sumArraysGPU这个核函数后会立即返回主机线程，所以为了计算出这个核函数花了多长时间，必须加上同步函数
    iElaps = cpuSecond() - iStart;
    printf("time is: %f\n", iElaps);

    // res_d -> res_from_gpu_h
    checkcuda(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));

    // cpu
    SumArrays(a_h, b_h, res_h, nElem);

    

    // check result
    check_res(res_h, res_from_gpu_h, (nElem * nElem)) ? printf("ok\n") : printf("error\n");

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    printf("Finish\n");

    return 0;
}