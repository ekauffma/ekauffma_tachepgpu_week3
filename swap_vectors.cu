#include <stdio.h>


const int DSIZE = 50;
const int block_size = 5;
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float *A, float *B, float *C, int v_size) {

    // Express the vector index in terms of threads and blocks
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Swap the vector elements - make sure you are not out of range
    if (idx < v_size) {
        C[idx] = A[idx]; // store A[idx] so that we can swap without losing info

        //perform swap
        A[idx] = B[idx];
        B[idx] = C[idx];
    }

}


int main() {


    float *d_A, *d_B, *d_C;
    float *h_A = (float*)malloc(DSIZE*sizeof(float));
    float *h_B = (float*)malloc(DSIZE*sizeof(float));
    float* h_C = (float*)malloc(DSIZE*sizeof(float));


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
    }

    printf("Before Swap:\n");
    printf("h_A = [");
    for (int i=0; i < DSIZE; i++) {
        printf("%f, ",h_A[i]);
    }
    printf("]\n");
    printf("h_B = [");
    for (int i=0; i < DSIZE; i++) {
        printf("%f, ",h_B[i]);
    }
    printf("]\n\n");

    // Allocate memory for host and device pointers 
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    vector_swap<<<grid_size, block_size>>>(d_A, d_B, d_C, DSIZE);

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure swapping was successful
    printf("After Swap:\n");
    printf("h_A = [");
    for (int i=0; i < DSIZE; i++) {
        printf("%f, ",h_A[i]);
    }
    printf("]\n");
    printf("h_B = [");
    for (int i=0; i < DSIZE; i++) {
        printf("%f, ",h_B[i]);
    }
    printf("]\n");

    // Free the memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
