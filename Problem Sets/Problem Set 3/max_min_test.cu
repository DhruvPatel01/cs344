#include "utils.h"

void min_or_max_driver(const float* const d_array, 
                       float* h_out, const size_t numElems,
                       bool is_max);

int main(int argc, char** argv) {
    int N = 1030;
    float h_array[N];
    float h_min, h_max;
    for (int i = 0; i < N; i++) 
        h_array[i] = i;
    float *d_array;

    checkCudaErrors(cudaMalloc((void **) &d_array, N*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_array, h_array, N, cudaMemcpyDeviceToHost));

    min_or_max_driver(d_array, &h_min, N, false);
    min_or_max_driver(d_array, &h_max, N, true);

    checkCudaErrors(cudaFree(d_array));
}