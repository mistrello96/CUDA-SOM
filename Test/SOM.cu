#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void
test_kernel(float* matrix, int numberNeuron, float* sample, float* distance, int sampleLength)
{
	//dovrei avere blocchetti da 14 valori, a ogni blocchetto corrisponde una "distanza"
	//sample in input. Chi la calcola? Chi la salva?
	int index = (threadIdx.x + blockIdx.x * blockDim.x);
	if (index < numberNeuron)
	{
		int matrixindex = index * 14;
		float tmp = 0;
		for(int i = 0; i < sampleLength; i++)
		{
			tmp = tmp + matrix[matrixindex+i] - sample[i];
		}

		distance[index] = distance[index] + tmp;
	}
}

int
main(void)
{

    int numberElements = 14;
    int numberRows = 2;
    int numberColoumns = 2;
    int numberNeuron = numberRows * numberColoumns;
    int totalLength = numberRows * numberColoumns * numberElements;

    float *hostMatrix = (float *)malloc(sizeof(float) * totalLength);
    float *hostSample = (float *)malloc(sizeof(float) * numberElements);
    float *hostDistance = (float *) malloc(sizeof(float) * numberNeuron);
    //random initialization
    for(int i = 0; i < totalLength; i++){
    	hostMatrix[i] = i;
    }
    for(int i = 0; i < numberNeuron; i++){
    	hostDistance[i] = 0;
    }
    for(int i = 0; i < numberElements; i++){
    	hostSample[i] = i+1;
    }

    float *deviceMatrix;
    float *deviceSample;
    float *deviceDistance;
    cudaMalloc((void **)&deviceMatrix, sizeof(float) * totalLength);
    cudaMalloc((void**)&deviceSample, sizeof(float) * numberElements);
    cudaMalloc((void**)&deviceDistance, sizeof(float) * numberNeuron);
	//copy to device memory
	cudaMemcpy(deviceMatrix, hostMatrix, sizeof(float) * totalLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSample, hostSample, sizeof(float) * numberElements, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDistance, hostDistance, sizeof(float) * numberNeuron, cudaMemcpyHostToDevice);	
	test_kernel<<<100,100>>>(deviceMatrix, numberNeuron, deviceSample, deviceDistance, numberElements);

    cudaDeviceSynchronize();

	cudaMemcpy(hostDistance, deviceDistance, sizeof(float) * numberNeuron, cudaMemcpyDeviceToHost);

    for(int i = 0; i < numberNeuron; i++){
    	std::cout << hostDistance[i] << std::endl;
    }

    cudaFree(deviceMatrix);
    free(hostMatrix);

}

