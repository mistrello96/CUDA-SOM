#include "distance_kernels.h"

// kernel used to compute euclidean distance between each neuron and the selected sample
__global__ void compute_distance_euclidean(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// tmpDistance will store the distances of the neuron's component
		double tmpDistance = 0;
		// elementDistance will store the distance of a single component
		double elementDistance = 0;
		for(int i = 0; i < nElements; i++)
		{
			elementDistance = k_matrix[index * nElements+i] - k_Samples[currentIndex + i];
			tmpDistance += elementDistance * elementDistance;
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = sqrt(tmpDistance);
	}
}

// kernel used to compute sum of squares distance between each neuron and the selected sample
__global__ void compute_distance_sum_squares(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// tmpDistance will store the distances of the neuron's component
		double tmpDistance = 0;
		// elementDistance will store the distance of a single component
		double elementDistance = 0;
		for(int i = 0; i < nElements; i++)
		{
			elementDistance = k_matrix[index * nElements+i] - k_Samples[currentIndex + i];
			tmpDistance += elementDistance * elementDistance;
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = tmpDistance;
	}
}

// kernel used to compute manhattan distance between each neuron and the selected sample
__global__ void compute_distance_manhattan(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// tmpDistance will store the distances of the neuron's component
		double tmpDistance = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmpDistance += fabs(k_matrix[index * nElements+i] - k_Samples[currentIndex + i]);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = tmpDistance;
	}
}

// kernel used to compute tanimoto distance between each neuron and the selected sample
__global__ void compute_distance_tanimoto(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		double crossproduct = 0;
		double norm1 = 0;
		double norm2 = 0;
		for(int i = 0; i < nElements; i++)
		{
			crossproduct += (k_matrix[index * nElements+i] * k_Samples[currentIndex + i]);
			norm1 += (k_matrix[index * nElements+i] * k_matrix[index * nElements+i]);
			norm2 += (k_Samples[currentIndex + i] * k_Samples[currentIndex + i]);

		}
		crossproduct = fabs(crossproduct);
		// save the distance of the neuron in distance vector
		k_distance[index] = crossproduct / (norm1+norm2-crossproduct);
	}
}