#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

/*
实现两个二维矩阵相加，使用2D Gird和2D Block
*/

// 通过CPU来计时
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

// 检查哪里报错
cudaError_t checkcuda(cudaError_t result)
{
    if(result != cudaSuccess)
    {
        fprintf(stderr, "Cuda runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// CPU版本的矩阵相加函数
void SumArrays(float *a, float *b, float *res, const int nx, const int ny)
{
    for(int j = 0; j < ny; j++)
    {
        for(int i = 0; i < nx; i++)
        {
            int idx = i + nx * j;
            res[idx] = a[idx] + b[idx];
        }
    }

}

// GPU版本的矩阵相加函数
__global__ void sumArraysGPU(float *a, float *b, float *res, const int nx, const int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix + iy * nx;

    // 这里防止数据溢出
    if(ix < nx && iy < ny)
    {
        res[idx] = a[idx] + b[idx];
    }
}

// 检查CPU与GPU运算结果是否完全相同
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
    //矩阵的大小
    int nx = 1 << 12; 
    int ny = 1 << 12;
    printf("Vector.size: %d\n", nx * ny);

    size_t nByte = nx * ny * sizeof(float);
    
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
    for(int y = 0; y < ny; y++)
    {
        for(int x = 0; x < nx; x++)
        {
            a_h[y * nx + x] = y;
            b_h[y * nx + x] = x;
        }
    }

    checkcuda(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    checkcuda(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));
    
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // gpu
    double iStart, iElaps;
    iStart = cpuSecond();
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nx, ny);
    cudaDeviceSynchronize(); // 这里如果不加同步，调用sumArraysGPU这个核函数后会立即返回主机线程，所以为了计算出这个核函数花了多长时间，必须加上同步函数
    iElaps = cpuSecond() - iStart;
    printf("time is: %f\n", iElaps);

    // res_d -> res_from_gpu_h
    checkcuda(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));

    // cpu
    SumArrays(a_h, b_h, res_h, nx, ny);

    // check result
    check_res(res_h, res_from_gpu_h, (nx * ny)) ? printf("ok\n") : printf("error\n");

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
