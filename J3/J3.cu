#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>

_global__ 
void KernelMat_Add(float *M,float*N,float *P,int dim){

    int row = blockDim.y*blockIdx.y +threadIdx.y;
    int column = blockDim.x*blockIdx.x +threadIdx.x;


    if((row<dim)&&(column<dim)){
        P[row*dim + column] = M[row*dim + column] + N[row*dim + column];
    }


}

void MatADD(float *M_h,float *N_h,float *P_h,int n){
    int size = n*n*sizeof(float);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void**)&M_d,size);
    cudaMalloc((void**)&N_d,size);
    cudaMalloc((void**)&P_d,size);

    cudaMemcpy(M_d,M_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(N_d,N_h,size,cudaMemcpyHostToDevice);

    dim3 dimGrid((n + 15) / 16, (n + 15) / 16, 1);
    dim3 dimBlock(16,16,1);

    KernelMat_Add <<<dimGrid,dimBlock>>>(M_d, N_d, P_d,n);

    cudaMemcpy(P_h,P_d,size,cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

}