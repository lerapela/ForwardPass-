#ifndef __ALL_FUNCTIONS__
#define __ALL_FUNCTIONS__

#include <cuda.h>
#include "BaseTypes.h"

__global__ void vMatrixMultiplication(F4* pf4MatrixA, F4* pf4MatrixB, F4* pf4MatrixC
            ,I4 i4MatrixA_rows, I4 i4MatrixA_cols, I4 i4MatrixB_cols);

__global__ void vReLU(F4* pf4Data, I4 i4Size);

__global__ void vSoftmax(F4* pf4InputData, F4* pf4OutputData, I4 i4BatchSize, I4 i4NumClasses);


__global__ void vBias(F4* pf4MatrixC, F4* pf4Bias, I4 i4MatrixA_rows, I4 i4MatrixB_cols);

#endif