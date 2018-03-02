#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void
test_kernel(float* matrix, int numElements)
{

	matrix[blockIdx.x] = matrix[blockIdx.x] * 2;
}

int
main(void)
{

    int numberElements = 14;
    int numberRows = 10;
    int numberColoumns = 10;
    int totalLength = numberRows * numberColoumns * numberElements;

    float *hostMatrix = (float *)malloc(sizeof(float) * totalLength);
    for(int i = 0; i < totalLength; i++){
    	hostMatrix[i] = i;
    	std::cout << hostMatrix[i] << std::endl;
    }

    float *deviceMatrix;
    cudaMalloc((void **)&deviceMatrix, sizeof(float) * totalLength);

	//copy to device memory
	cudaMemcpy(deviceMatrix, hostMatrix, sizeof(float) * totalLength, cudaMemcpyHostToDevice);

	test_kernel<<<totalLength,1>>>(deviceMatrix, totalLength);

    cudaDeviceSynchronize();

	cudaMemcpy(hostMatrix, deviceMatrix, sizeof(float) * totalLength, cudaMemcpyDeviceToHost);

    for(int i = 0; i < totalLength; i++){
    	std::cout << hostMatrix[i] << std::endl;
    }

    cudaFree(deviceMatrix);
    free(hostMatrix);

}

