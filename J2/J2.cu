#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>

__global__ 
void KernelAdd(float *A,float *B,float *C,int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        C[i]= A[i]+B[i];
    }
}

void VecADD(float *A_h,float *B_h,float *C_h,int n){
    int size = n*sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d,size);
    cudaMalloc((void**)&B_d,size);
    cudaMalloc((void**)&C_d,size);

    cudaMemcpy(A_d,A_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,size,cudaMemcpyHostToDevice);

    KernelAdd <<<ceil(n/256.0),256>>>(A_d, B_d, C_d,n);

    cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

int main() {

int n = 5;

float A[] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
float B[] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
float C[5];

VecADD(A,B,C,n);

 printf("Resultats:\n");
int i;
for(i = 0; i < n; i++) {
    printf("%f + %f = %f\n", A[i], B[i], C[i]);
    }

     return 0;
}