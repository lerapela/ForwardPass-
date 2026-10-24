#include "Includes.h"


__global__ void vMatrixMultiplication(F4* pf4MatrixA, F4* pf4MatrixB, F4* pf4MatrixC
            ,I4 i4MatrixA_rows, I4 i4MatrixA_cols, I4 i4MatrixB_cols)
{
    I4 i4rows = blockDim.y * blockIdx.y + threadIdx.y;
    I4 i4cols = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i4rows < i4MatrixA_rows && i4cols < i4MatrixB_cols){

        F4 f4Sum = 0.0f;
        for(I4 i4idx = 0; i4idx < i4MatrixA_cols; i4idx++)
        {
            f4Sum += pf4MatrixA[i4rows * i4MatrixA_cols + i4idx] * pf4MatrixB[i4idx * i4MatrixB_cols + i4cols];
        }

        pf4MatrixC[i4rows * i4MatrixB_cols + i4cols] = f4Sum;
    }

}