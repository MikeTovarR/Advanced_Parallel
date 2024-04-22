#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__device__ int isValid(const int* board, int row, int col, int num) {
    // Check row
    for (int i = 0; i < 9; ++i) {
        if (board[row * 9 + i] == num) return 0;
    }

    // Check column
    for (int i = 0; i < 9; ++i) {
        if (board[i * 9 + col] == num) return 0;
    }

    // Check subgrid
    int startRow = row - row % 3;
    int startCol = col - col % 3;
    for (int i = startRow; i < startRow + 3; ++i) {
        for (int j = startCol; j < startCol + 3; ++j) {
            if (board[i * 9 + j] == num) return 0;
        }
    }

    return 1;
}

__global__ void solutionKernel(int* board, int* found) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (*found || idx >= 81) return;

    int row = idx / 9;
    int col = idx % 9;

    if (board[idx] == 0) {
        for (int num = 1; num <= 9; ++num) {
            if (isValid(board, row, col, num)) {
                board[idx] = num;
                if (idx == 80) {
                    *found = 1;
                    return;
                }
            }
        }
    }
    else {
        if (idx == 80) {
            *found = 1;
            return;
        }
    }
}

int main() {
    // Definir el tablero de Sudoku en el host
    int host_board[9][9] = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

    printf("Initial board:\n");
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            printf("%d ", host_board[i][j]);
        }
        printf("\n");
    }

    // Copiar el tablero de Sudoku al dispositivo
    int* device_board;
    cudaMalloc((void**)&device_board, sizeof(int) * 81);
    cudaMemcpy(device_board, host_board, sizeof(int) * 81, cudaMemcpyHostToDevice);

    // Inicializar variable para indicar si se encontró una solución en el dispositivo
    int* device_found;
    cudaMalloc((void**)&device_found, sizeof(int));
    int host_found = 0;

    // Definición de la configuración de lanzamiento del kernel
    int numBlocks = 9; // Número de bloques
    int threadsPerBlock = 9; // Número de hilos por bloque

    // Lanzar el kernel para resolver el Sudoku en paralelo
    solutionKernel << <numBlocks, threadsPerBlock >> > (device_board, device_found);

    // Copiar el resultado de vuelta al host
    cudaMemcpy(&host_found, device_found, sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir el resultado
    if (host_found) {
        int solved_board[81];
        cudaMemcpy(solved_board, device_board, sizeof(int) * 81, cudaMemcpyDeviceToHost);
        printf("Solution found:\n");
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                printf("%d ", solved_board[i * 9 + j]);
            }
            printf("\n");
        }
    }
    else {
        printf("No solution exists for the given Sudoku board.\n");
    }

    // Liberar memoria en el dispositivo
    cudaFree(device_board);
    cudaFree(device_found);

    return 0;
}
