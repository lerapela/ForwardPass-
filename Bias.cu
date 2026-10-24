#include "Includes.h"

__global__ void vBias(F4* pf4MatrixC, F4* pf4Bias, I4 i4MatrixA_rows, I4 i4MatrixB_cols)
{
    I4 i4Idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(i4Idx < (i4MatrixA_rows * i4MatrixB_cols))
    {
        pf4MatrixC[i4Idx] += pf4Bias[i4Idx % i4MatrixB_cols];
    }
}