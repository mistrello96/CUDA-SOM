// kernel to find the distance of each neuron from the sample vector
__global__ void compute_distance_euclidean_normalized(double* k_matrix, double* k_ActualSample, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// computing the corresponding index in the matrix
		int matrixindex = index * nElements;
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + powf(k_matrix[matrixindex+i] - k_ActualSample[i], 2.0);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = sqrtf(tmp)/nElements;
	}
}

__global__ void compute_distance_euclidean(double* k_matrix, double* k_ActualSample, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// computing the corresponding index in the matrix
		int matrixindex = index * nElements;
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + powf(k_matrix[matrixindex+i] - k_ActualSample[i], 2.0);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = sqrtf(tmp)/nElements;
	}
}

__global__ void compute_distance_sum_squares(double* k_matrix, double* k_ActualSample, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// computing the corresponding index in the matrix
		int matrixindex = index * nElements;
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + powf(k_matrix[matrixindex+i] - k_ActualSample[i], 2.0);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = tmp;
	}
}

__global__ void compute_distance_sum_squares_normalized(double* k_matrix, double* k_ActualSample, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// computing the corresponding index in the matrix
		int matrixindex = index * nElements;
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + powf(k_matrix[matrixindex+i] - k_ActualSample[i], 2.0);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = tmp/nElements;
	}
}

__global__ void compute_distance_manhattan_(double* k_matrix, double* k_ActualSample, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// computing the corresponding index in the matrix
		int matrixindex = index * nElements;
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + fabs(k_matrix[matrixindex+i] - k_ActualSample[i]);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = tmp;
	}
}

__global__ void compute_distance_manhattan_normalized(double* k_matrix, double* k_ActualSample, double* k_distance, int nNeuron, int nElements)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// computing the corresponding index in the matrix
		int matrixindex = index * nElements;
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + fabs(k_matrix[matrixindex+i] - k_ActualSample[i]);
		}

		// save the distance of the neuron in distance vector
		k_distance[index] = tmp/nElements;
	}
}