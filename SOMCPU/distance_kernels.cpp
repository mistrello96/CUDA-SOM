// kernel used to compute euclidean distance between each neuron and the selected sample
void compute_distance_euclidean(double* h_matrix, double* h_Samples, int currentIndex, double* h_distance, int nNeuron, int nElements)
{
	for (int index = 0; index < nNeuron; index++)
	{
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + (h_matrix[index * nElements+i] - h_Samples[currentIndex + i]) * (h_matrix[index * nElements+i] - h_Samples[currentIndex + i]);
		}

		// save the distance of the neuron in distance vector
		h_distance[index] = sqrt(tmp);
	}
}

// kernel used to compute sum of squares distance between each neuron and the selected sample
void compute_distance_sum_squares(double* h_matrix, double* h_Samples, int currentIndex, double* h_distance, int nNeuron, int nElements)
{
	for (int index = 0; index < nNeuron; index++)
	{
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + (h_matrix[index * nElements+i] - h_Samples[currentIndex + i]) * (h_matrix[index * nElements+i] - h_Samples[currentIndex + i]);
		}

		// save the distance of the neuron in distance vector
		h_distance[index] = tmp;
	}
}

// kernel used to compute manhattan distance between each neuron and the selected sample
void compute_distance_manhattan(double* h_matrix, double* h_Samples, int currentIndex, double* h_distance, int nNeuron, int nElements)
{
	for (int index = 0; index < nNeuron; index++)
	{
		// tmp will store the distances of the neuron's component
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + fabs(h_matrix[index * nElements+i] - h_Samples[currentIndex + i]);
		}

		// save the distance of the neuron in distance vector
		h_distance[index] = tmp;
	}
}

// kernel used to compute tanimoto distance between each neuron and the selected sample
void compute_distance_tanimoto(double* h_matrix, double* h_Samples, int currentIndex, double* h_distance, int nNeuron, int nElements)
{
	for (int index = 0; index < nNeuron; index++)
	{
		double crossproduct = 0;
		double norm1 = 0;
		double norm2 = 0;
		for(int i = 0; i < nElements; i++)
		{
			crossproduct = crossproduct + (h_matrix[index * nElements+i] * h_Samples[currentIndex + i]);
			norm1 = norm1 + (h_matrix[index * nElements+i] * h_matrix[index * nElements+i]);
			norm2 = norm2 + (h_Samples[currentIndex + i] * h_Samples[currentIndex + i]);

		}
		crossproduct = fabs(crossproduct);
		// save the distance of the neuron in distance vector
		h_distance[index] = crossproduct / (norm1+norm2-crossproduct);
	}
}