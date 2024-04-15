#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define GPUErrorAssertion(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void convolution2D(int* mat, int* res, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width && row < height) {
        int g_index = row * width + col;
        float sum = 0.0f;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int curRow = row + i;
                int curCol = col + j;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    sum += mat[curRow * width + curCol];
                }
            }
        }
        res[g_index] = sum;
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int dataSize = width * height * sizeof(int);

    int* M, * M_res;
    int* M_gpu, * M_res_gpu;

    M = (int*)malloc(dataSize);
    M_res = (int*)malloc(dataSize);

    GPUErrorAssertion(cudaMalloc((void**)&M_gpu, dataSize));
    GPUErrorAssertion(cudaMalloc((void**)&M_res_gpu, dataSize));

    for (int i = 0; i < width * height; ++i) {
        M[i] = rand() % 9;
    }

    printf("Before: \n");
    for (int i = 0; i < 20; ++i) {
        printf("M[%d] = %d\n", i, M[i]);

    }

    GPUErrorAssertion(cudaMemcpy(M_gpu, M, dataSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    convolution2D << <gridSize, blockSize >> > (M_gpu, M_res_gpu, width, height);
    GPUErrorAssertion(cudaDeviceSynchronize());

    GPUErrorAssertion(cudaMemcpy(M_res, M_res_gpu, dataSize, cudaMemcpyDeviceToHost));

    printf("After: \n");
    for (int i = 0; i < 20; ++i) {
        printf("RES[%d] = %d\n", i, M_res[i]);
    }

    cudaFree(M_gpu);
    cudaFree(M_res_gpu);
    free(M);
    free(M_res);

    return 0;
}