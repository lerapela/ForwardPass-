#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "Includes.h"


#define INPUT_SIZE 784       // 28 * 28 pixels
#define HIDDEN_SIZE 128      // The hidden layer size 
#define OUTPUT_SIZE 10       // Digits 0 through 9
#define BATCH_SIZE 10000    // No of images processed

using namespace std;

void vloadBinaryFile(string strFileName, F4** pf4ToFile, I4 i4Size1, I4 i4Size2) 
{

    FILE *file =  fopen(strFileName.c_str(), "rb");

    if(!file)
    {
        fprintf(stderr, "Failed to open %s binary file\n", strFileName.c_str());
        exit(-1);
    }
    cudaMallocManaged(pf4ToFile, sizeof(F4) * i4Size1 * i4Size2);
    fread(*pf4ToFile,  sizeof(F4), i4Size1 * i4Size2, file);

    fclose(file);

}

void vKernelError()
{
    cudaError_t clError = cudaGetLastError();
    if(clError != cudaSuccess)
    {
        printf("Kernel Error: %s\n", cudaGetErrorString(clError));
        exit(-1);
    }

}

int main()
{

    // Grid and threadBlocks size defs
    dim3 blockDim(16, 16);
    dim3 gridDim((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                 (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

    dim3 gridDim2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

    I4 i4N1 = BATCH_SIZE * HIDDEN_SIZE;
    I4 i4N2 = BATCH_SIZE * OUTPUT_SIZE;

    string strInputDataName = "./BinaryFiles/x_test.bin";
    string strW1Name        = "./BinaryFiles/W1.bin";
    string strW2Name        = "./BinaryFiles/W2.bin";
    string strBias1Name     = "./BinaryFiles/b1.bin";
    string strBias2Name     = "./BinaryFiles/b2.bin";

    // All pointers
    F4* pf4Input;
    F4* pf4Weights1;
    F4* pf4Weights2;
    F4* pf4Bias1;
    F4* pf4Bias2;
    F4* pf4Output;
    F4* pf4HiddenOutput;
    F4* pf4finalOutput;

    // Allocate and populate memory
    vloadBinaryFile(strInputDataName, &pf4Input, BATCH_SIZE, INPUT_SIZE);
    vloadBinaryFile(strW1Name, &pf4Weights1, INPUT_SIZE, HIDDEN_SIZE);
    vloadBinaryFile(strW2Name, &pf4Weights2, OUTPUT_SIZE, HIDDEN_SIZE);
    vloadBinaryFile(strBias1Name, &pf4Bias1, HIDDEN_SIZE, 1);
    vloadBinaryFile(strBias2Name, &pf4Bias2, OUTPUT_SIZE, 1);


    cudaMallocManaged(&pf4Output, sizeof(F4) * BATCH_SIZE * HIDDEN_SIZE);                     // Stores first matix mul output
    cudaMallocManaged(&pf4HiddenOutput, sizeof(F4) * BATCH_SIZE * HIDDEN_SIZE);               // Stores second matrix mul output
    cudaMallocManaged(&pf4finalOutput, sizeof(F4) * BATCH_SIZE * HIDDEN_SIZE);                 // Stores final output from softmax  

    for (I4 i4idx = 0; i4idx < BATCH_SIZE * INPUT_SIZE; i4idx++) pf4Input[i4idx] /= 255.0f;   // Normalize the inputData


    // Kernels executions
    vMatrixMultiplication<<<gridDim, blockDim>>>(pf4Input, pf4Weights1, pf4Output, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
    cudaDeviceSynchronize();
    vKernelError();

    vBias<<<(i4N1 + 255) / 256, 256>>>(pf4Output, pf4Bias1, BATCH_SIZE, HIDDEN_SIZE);
    cudaDeviceSynchronize();
    vKernelError();

    vReLU<<<(i4N1 + 255) / 256, 256>>>(pf4Output, BATCH_SIZE * HIDDEN_SIZE);
     cudaDeviceSynchronize();
    vKernelError();
    
    vMatrixMultiplication<<<gridDim2, blockDim>>>(pf4Output, pf4Weights2, pf4HiddenOutput, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    cudaDeviceSynchronize();
    vKernelError();

    vBias<<<( i4N2 + 255) / 256, 256>>>(pf4HiddenOutput, pf4Bias2, BATCH_SIZE, OUTPUT_SIZE);
    cudaDeviceSynchronize();
    vKernelError();

    vSoftmax<<<(BATCH_SIZE + 255) / 256, 256>>>(pf4HiddenOutput, pf4finalOutput, BATCH_SIZE, OUTPUT_SIZE);
    cudaDeviceSynchronize();
    vKernelError();




    // Probability predictions on first 5 images [InputData]
    for (I4 i4Idx = 0; i4Idx < 5; i4Idx++)
    {
        printf("Image %d: ", i4Idx);
        for (I4 i4Idy = 0; i4Idy < OUTPUT_SIZE; i4Idy++)
            printf("%.4f ", pf4finalOutput[i4Idx * OUTPUT_SIZE + i4Idy]);
            printf("\n");
    }


    //=================Evaluation====================

    I4* pi4Labels = (I4*)malloc(BATCH_SIZE * sizeof(I4));

    FILE* fLabels = fopen("./BinaryFiles/y_test.bin", "rb");
    if (!fLabels)
    { 
        perror("Failed to open the test file"); 
        return 1;
    }
    size_t labels_read = fread(pi4Labels, sizeof(I4), BATCH_SIZE, fLabels);
    fclose(fLabels);
    printf("Loaded %zu labels\n", labels_read);

    I4 i4Correct = 0;
    for (I4 i4Idx = 0; i4Idx < BATCH_SIZE; i4Idx++) {
        I4 i4Pred = 0;
        for (I4 i4Idy = 0; i4Idy < OUTPUT_SIZE; i4Idy++)
        {
            if (pf4finalOutput[i4Idx * OUTPUT_SIZE + i4Idy] > pf4finalOutput[i4Idx * OUTPUT_SIZE + i4Pred])
            {
                i4Pred = i4Idy;
            }
        }
        if (i4Pred == pi4Labels[i4Idx])
        {
            i4Correct++;
        } 
    }
    printf("Accuracy: %d / %d\n", i4Correct, BATCH_SIZE);
    


    cudaFree(pf4Input);
    cudaFree(pf4Weights1);
    cudaFree(pf4Weights2);
    cudaFree(pf4Output);
    cudaFree(pf4HiddenOutput);
    cudaFree(pf4finalOutput);
    cudaFree(pf4Bias1);
    cudaFree(pf4Bias2);

    return 0;
}


