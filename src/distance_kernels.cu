__global__ void compute_distance(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + (k_matrix[index * nElements+i] - k_Samples[currentIndex + i]) * (k_matrix[index * nElements+i] - k_Samples[currentIndex + i]);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = sqrt(tmp);
	}
}

__global__ void compute_distance_manhattan(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + fabs(k_matrix[index * nElements+i] - k_Samples[currentIndex + i]);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = tmp;
	}
}

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
			crossproduct = crossproduct + (k_matrix[index * nElements+i] * k_Samples[currentIndex + i]);
			norm1 = norm1 + (k_matrix[index * nElements+i] * k_matrix[index * nElements+i]);
			norm2 = norm2 + (k_Samples[currentIndex + i] * k_Samples[currentIndex + i]);

		}
		crossproduct = fabs(crossproduct);
		// save the distance of the neuron in distance vector
		k_distance[index] = crossproduct / (norm1+norm2-crossproduct);
	}
}