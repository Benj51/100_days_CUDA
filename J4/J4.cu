#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>


__global__ 
void Kernel_MatMul(float *M_h,float *N_h,float *P_h,int n){

    int row = blockDim.y * blockIdx.y +threadIdx.y;
    int column = blockDim.x * blockIdx.x +threadIdx.x;

    if((row<n)&&(column<n)){
        float dot_product = 0.0f;
        for(int k = 0; k<n; k++){
            dot_product += M_h[row*k+column] * N_h[row*k+column];      
        }
        P_h[row*n+column]= dot_product;
    }

}

void MatMul(float *M_h,float *N_h,float *P_h,int n){
    int size = n*n*sizeof(float);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void**)&M_d,size);
    cudaMalloc((void**)&N_d,size);
    cudaMalloc((void**)&P_d,size);

    cudaMemcpy(M_d,M_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(N_d,N_h,size,cudaMemcpyHostToDevice);

    dim3 dimGrid((n + 15) / 16, (n + 15) / 16, 1);
    dim3 dimBlock(16,16,1);

   Kernel_MatMul <<<dimGrid,dimBlock>>>(M_d, N_d, P_d,n);

    cudaMemcpy(P_h,P_d,size,cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

}
