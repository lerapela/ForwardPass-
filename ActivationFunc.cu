#include <math.h>
#include "Includes.h"


// Activation function: ReLU => which just returns a value greater between 0 and x (ReLU(x) = max(0, x))

__global__ void vReLU(F4* pf4Data, I4 i4Size)
{
    I4 i4Idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i4Idx < i4Size)
    {
        pf4Data[i4Idx] = fmaxf(0.0f, pf4Data[i4Idx]);
    }
}

// Activation function: Softmax => which takes input data and normalize it into a probability distribution of input data elements

__global__ void vSoftmax(F4* pf4InputData, F4* pf4OutputData, I4 i4BatchSize, I4 i4NumClasses)
{
    I4 i4Batch_indx = blockDim.x * blockIdx.x + threadIdx.x;

    if(i4Batch_indx < i4BatchSize)
    {
        I4 i4RowIndex = i4Batch_indx * i4NumClasses;
        F4 f4MaxVal   = pf4InputData[i4RowIndex + 0]; 

        for(I4 i4Idx = 0; i4Idx < i4NumClasses; i4Idx++)
        {   // Find maximum value for numerical stability
            f4MaxVal = fmaxf(f4MaxVal, pf4InputData[i4Batch_indx * i4NumClasses + i4Idx]); 
        }

        F4 f4Sum = 0.0f;
        for(I4 i4Idx = 0; i4Idx < i4NumClasses; i4Idx++){
    
            pf4OutputData[i4Batch_indx * i4NumClasses + i4Idx] = expf(pf4InputData[i4Batch_indx * i4NumClasses + i4Idx] - f4MaxVal);
            f4Sum += pf4OutputData[i4Batch_indx * i4NumClasses + i4Idx];
        }

        // Normalize
        for(I4 i4Idx = 0; i4Idx < i4NumClasses; i4Idx++)
        {
            pf4OutputData[i4Batch_indx * i4NumClasses + i4Idx] /= f4Sum;
        }
    }
}


