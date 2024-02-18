#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

__global__ void vectors_sum(const int* data_a, const int* data_b, const int* data_c, int* r) {

    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x + blockDim.y * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int blockGrid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    int idx = blockGrid * blockSize + tid;
    r[idx] = data_a[idx] + data_b[idx] + data_c[idx];
    printf("%d + %d + %d = %d\n", data_a[idx], data_b[idx], data_c[idx], r[idx]);
}

int main()
{

    dim3 block_size(4, 5, 5);
    dim3 grid_size(4, 5, 5);

    int* a_cpu;
    int* b_cpu;
    int* c_cpu;

    int* a_device;
    int* b_device;
    int* c_device;

    int* r_device;

    // Memory allocation data
    const int vector_len = 10000;
    const int data_size = sizeof(int) * vector_len;

    a_cpu = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);
    c_cpu = (int*)malloc(data_size);

    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&r_device, data_size);

    for (int i = 0; i < vector_len; i++) {
        a_cpu[i] = i;
        b_cpu[i] = i;
        c_cpu[i] = i;
    }

    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);
    
    vectors_sum << <grid_size, block_size >> > (a_device, b_device, c_device, r_device);

    cudaDeviceReset();
    cudaFree(c_device);
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(r_device);

    free(a_cpu);
    free(b_cpu);
    free(c_cpu);

    return 0;
}