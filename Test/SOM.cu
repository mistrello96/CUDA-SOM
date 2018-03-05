#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

//BUG, se aumento blocchi, non mi calcola più correttamente le distanze ma lascia 0

__global__ void
test_kernel(float* matrix, int numberNeuron, float* sample, float* distance, int sampleLength)
{
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
	// number of features in each neuron
    int numberElements = 14;
    // number of rows in the martix
    int numberRows = 100;
    // number of column in the martix
    int numberColoumns = 100;
    // total number of neurons in the SOM
    int numberNeuron = numberRows * numberColoumns;
    // total length of the serialized matrix
    int totalLength = numberRows * numberColoumns * numberElements;

    // host SOM
    float *hostMatrix = (float *)malloc(sizeof(float) * totalLength);
    // host sample array
    float *hostSample = (float *)malloc(sizeof(float) * numberElements);
    // host distance array, used to find BMU
    float *hostDistance = (float *) malloc(sizeof(float) * numberNeuron);

    //random SOM initialization
    for(int i = 0; i < totalLength; i++){
    	hostMatrix[i] = 5;
    }
    // distance array inizialization
    for(int i = 0; i < numberNeuron; i++){
    	hostDistance[i] = 0;
    }

    //random sample inizialization, used for TEST
    for(int i = 0; i < numberElements; i++){
    	hostSample[i] = i+1;
    }

    // device SOM
    float *deviceMatrix;
    // device sample array
    float *deviceSample;
    // device distance array, 
    float *deviceDistance;

    //device malloc
    cudaMalloc((void **)&deviceMatrix, sizeof(float) * totalLength);
    cudaMalloc((void**)&deviceSample, sizeof(float) * numberElements);
    cudaMalloc((void**)&deviceDistance, sizeof(float) * numberNeuron);

	//copy from host to device matrix, sample and distance
	cudaMemcpy(deviceMatrix, hostMatrix, sizeof(float) * totalLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSample, hostSample, sizeof(float) * numberElements, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDistance, hostDistance, sizeof(float) * numberNeuron, cudaMemcpyHostToDevice);	
	
    //peparing param to launch kernel
    int nblocks = (numberNeuron % 1024) + 1; 
    test_kernel<<<nblocks,1024>>>(deviceMatrix, numberNeuron, deviceSample, deviceDistance, numberElements);

	//wait for all block to be completed
    cudaDeviceSynchronize();

    /*
    cudaMemcpy(hostDistance, deviceDistance, sizeof(float) * numberNeuron, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < numberNeuron; i++){
        std::cout << hostDistance[i] << std::endl;
    }
    */

	//create thrust vector to find BMU
	thrust::device_vector<float> d_vec(deviceDistance, deviceDistance + numberNeuron);
	//extract the first element
	thrust::device_vector<float>::iterator iter = thrust::min_element(d_vec.begin(), d_vec.end());
	// find index of BMU
	unsigned int position = iter - d_vec.begin();
	float min_value = *iter;

	std::cout << "The minimum value is " << min_value << " at position " << position << std::endl;
	//TODO: update BMU and neighbors
 

    cudaFree(deviceMatrix);
    free(hostMatrix);

}

