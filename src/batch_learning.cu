#include "batch_learning.h"
#include "utility_functions.h"
#include <stdio.h>

#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ < 600
	__device__ double atomicAdd(double* address, double val)
	{
	    unsigned long long int* address_as_ull =
	                              (unsigned long long int*)address;
	    unsigned long long int old = *address_as_ull, assumed;

	    do {
	        assumed = old;
	        old = atomicCAS(address_as_ull, assumed,
	                        __double_as_longlong(val +
	                               __longlong_as_double(assumed)));

	    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	    } while (assumed != old);

	    return __longlong_as_double(old);
	}
	#endif
#endif

// kernel used to compute euclidean distance between each neuron and the selected sample
__global__ void batch_compute_distance_euclidean(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, double* k_Samples, double* k_Distance, int * k_BMU, int nSamples, int nNeuron, int nElements, bool normalizedistance, char neighborsType, int nRows, int nColumns, bool toroidal, int radius, char lattice)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nSamples)
	{
		// SEARCH OF BMU
		int BMUindex = 0;
		double BMUvalue = 1.79769e+308;
		// scanning all neurons to find BMU
		for (int i = 0; i < nNeuron; i++)
		{
			// tmpDistance will store the distances of the neuron's component
			double tmpDistance = 0;
			// elementDistance will store the distance of a single component
			double elementDistance = 0;
			for(int j = 0; j < nElements; j++)
			{
				elementDistance = k_Matrix[i * nElements + j] - k_Samples[index * nElements + j];
				tmpDistance += elementDistance * elementDistance;
			}
			tmpDistance = sqrt(tmpDistance);
			if (tmpDistance < BMUvalue)
			{
				BMUvalue = tmpDistance;
				BMUindex = i;
			}
			
		}

		// SAVE BMU INFO IN THE ARRAY
		k_BMU[index] = BMUindex;
		if(!normalizedistance)
		{
            k_Distance[index] = BMUvalue; 
		}
        else
        {
			k_Distance[index] = BMUvalue * BMUvalue;
        }

        // SAVE THE INFLUENCE OF THE SAMPLE ON THE NETWORK
		if (radius == 0)
		{
		    for (int i = BMUindex * nElements, j=0; j < nElements; i++, j++)
		    {
		        atomicAdd(&k_Matrix_num[i], k_Samples[index * nElements + j]); 
		    }
		    atomicAdd(&k_Matrix_denum[BMUindex], 1.0f);
		}
		else
        {
            if (toroidal)
            {
                if(lattice == 's')
                {
                    for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceToroidal(n / nColumns, n % nColumns, BMUindex / nColumns, BMUindex % nColumns, nRows, nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
                else
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceHexGridToroidal(BMUindex / nColumns, BMUindex % nColumns, n / nColumns, n % nColumns, nRows, nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
            }
            else
            {
                if(lattice == 's')
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = sqrtf(((n / nColumns) - (BMUindex / nColumns)) * ((n / nColumns) - (BMUindex / nColumns)) + ((n % nColumns) - (BMUindex % nColumns)) * ((n % nColumns) - (BMUindex % nColumns)));
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
                else
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceHexGrid(BMUindex / nColumns, BMUindex % nColumns, n / nColumns, n % nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
            }
        }
	}
}

// kernel used to compute sum of squares distance between each neuron and the selected sample
__global__ void batch_compute_distance_sum_squares(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, double* k_Samples, double* k_Distance, int * k_BMU, int nSamples, int nNeuron, int nElements, bool normalizedistance, char neighborsType, int nRows, int nColumns, bool toroidal, int radius, char lattice)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nSamples)
	{
		// SEARCH OF BMU
		int BMUindex = 0;
		double BMUvalue = 1.79769e+308;
		// scanning all neurons to find BMU
		for (int i = 0; i < nNeuron; i++)
		{
			// tmpDistance will store the distances of the neuron's component
			double tmpDistance = 0;
			// elementDistance will store the distance of a single component
			double elementDistance = 0;
			for(int j = 0; j < nElements; j++)
			{
				elementDistance = k_Matrix[i * nElements + j] - k_Samples[index * nElements + j];
				tmpDistance += elementDistance * elementDistance;
			}

			if (tmpDistance < BMUvalue){
				BMUvalue = tmpDistance;
				BMUindex = i;
			}
		}

		// SAVE BMU INFO IN THE ARRAY
		k_BMU[index] = BMUindex;
		if(!normalizedistance)
		{
            k_Distance[index] = BMUvalue; 
		}
        else
        {
			k_Distance[index] = BMUvalue * BMUvalue;
        }

        // SAVE THE INFLUENCE OF THE SAMPLE ON THE NETWORK
		if (radius == 0)
		{
		    for (int i = BMUindex * nElements, j=0; j < nElements; i++, j++)
		    {
		        atomicAdd(&k_Matrix_num[i], k_Samples[index * nElements + j]); 
		    }
		    atomicAdd(&k_Matrix_denum[BMUindex], 1.0f);
		}
		else
        {
            if (toroidal)
            {
                if(lattice == 's')
                {
                    for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceToroidal(n / nColumns, n % nColumns, BMUindex / nColumns, BMUindex % nColumns, nRows, nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
                else
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceHexGridToroidal(BMUindex / nColumns, BMUindex % nColumns, n / nColumns, n % nColumns, nRows, nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
            }
            else
            {
                if(lattice == 's')
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = sqrtf(((n / nColumns) - (BMUindex / nColumns)) * ((n / nColumns) - (BMUindex / nColumns)) + ((n % nColumns) - (BMUindex % nColumns)) * ((n % nColumns) - (BMUindex % nColumns)));
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
                else
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceHexGrid(BMUindex / nColumns, BMUindex % nColumns, n / nColumns, n % nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
            }
        }
	}
}

// kernel used to compute manhattan distance between each neuron and the selected sample
__global__ void batch_compute_distance_manhattan(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, double* k_Samples, double* k_Distance, int * k_BMU, int nSamples, int nNeuron, int nElements, bool normalizedistance, char neighborsType, int nRows, int nColumns, bool toroidal, int radius, char lattice)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nSamples)
	{
		// SEARCH OF BMU
		int BMUindex = 0;
		double BMUvalue = 1.79769e+308;
		// scanning all neurons to find BMU
		for (int i = 0; i < nNeuron; i++)
		{
			// tmpDistance will store the distances of the neuron's component
			double tmpDistance = 0;
			for(int j = 0; j < nElements; j++)
			{
				tmpDistance += fabs(k_Matrix[i * nElements + j] - k_Samples[index * nElements + j]);
			}

			if (tmpDistance < BMUvalue){
				BMUvalue = tmpDistance;
				BMUindex = i;
			}
		}

		// SAVE BMU INFO IN THE ARRAY
		k_BMU[index] = BMUindex;
		if(!normalizedistance)
		{
            k_Distance[index] = BMUvalue; 
		}
        else
        {
			k_Distance[index] = BMUvalue * BMUvalue;
        }

        // SAVE THE INFLUENCE OF THE SAMPLE ON THE NETWORK
		if (radius == 0)
		{
		    for (int i = BMUindex * nElements, j=0; j < nElements; i++, j++)
		    {
		        atomicAdd(&k_Matrix_num[i], k_Samples[index * nElements + j]); 
		    }
		    atomicAdd(&k_Matrix_denum[BMUindex], 1.0f);
		}
		else
        {
            if (toroidal)
            {
                if(lattice == 's')
                {
                    for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceToroidal(n / nColumns, n % nColumns, BMUindex / nColumns, BMUindex % nColumns, nRows, nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
                else
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceHexGridToroidal(BMUindex / nColumns, BMUindex % nColumns, n / nColumns, n % nColumns, nRows, nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
            }
            else
            {
                if(lattice == 's')
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = sqrtf(((n / nColumns) - (BMUindex / nColumns)) * ((n / nColumns) - (BMUindex / nColumns)) + ((n % nColumns) - (BMUindex % nColumns)) * ((n % nColumns) - (BMUindex % nColumns)));
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
                else
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceHexGrid(BMUindex / nColumns, BMUindex % nColumns, n / nColumns, n % nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
            }
        }
	}
}

// kernel used to compute tanimoto distance between each neuron and the selected sample
__global__ void batch_compute_distance_tanimoto(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, double* k_Samples, double* k_Distance, int * k_BMU, int nSamples, int nNeuron, int nElements, bool normalizedistance, char neighborsType, int nRows, int nColumns, bool toroidal, int radius, char lattice)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nSamples)
	{
		// SEARCH OF BMU
		int BMUindex = 0;
		double BMUvalue = 1.79769e+308;
		// scanning all neurons to find BMU
		for (int i = 0; i < nNeuron; i++)
		{
			double crossproduct = 0;
			double norm1 = 0;
			double norm2 = 0;
			for(int j = 0; j < nElements; j++)
			{
				crossproduct += (k_Matrix[i * nElements + j] * k_Samples[index * nElements + j]);
				norm1 += (k_Matrix[i * nElements + j] * k_Matrix[i * nElements + j]);
				norm2 += (k_Samples[index * nElements + j] * k_Samples[index * nElements + j]);
			}
			crossproduct = fabs(crossproduct);
			crossproduct = crossproduct / (norm1+norm2-crossproduct);
			if (crossproduct < BMUvalue){
				BMUvalue = crossproduct;
				BMUindex = i;
			}
		}

		// SAVE BMU INFO IN THE ARRAY
		k_BMU[index] = BMUindex;
		if(!normalizedistance)
		{
            k_Distance[index] = BMUvalue; 
		}
        else
        {
			k_Distance[index] = BMUvalue * BMUvalue;
        }

        // SAVE THE INFLUENCE OF THE SAMPLE ON THE NETWORK
		if (radius == 0)
		{
		    for (int i = BMUindex * nElements, j=0; j < nElements; i++, j++)
		    {
		        atomicAdd(&k_Matrix_num[i], k_Samples[index * nElements + j]); 
		    }
		    atomicAdd(&k_Matrix_denum[BMUindex], 1.0f);
		}
		else
        {
            if (toroidal)
            {
                if(lattice == 's')
                {
                    for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceToroidal(n / nColumns, n % nColumns, BMUindex / nColumns, BMUindex % nColumns, nRows, nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
                else
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceHexGridToroidal(BMUindex / nColumns, BMUindex % nColumns, n / nColumns, n % nColumns, nRows, nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
            }
            else
            {
                if(lattice == 's')
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = sqrtf(((n / nColumns) - (BMUindex / nColumns)) * ((n / nColumns) - (BMUindex / nColumns)) + ((n % nColumns) - (BMUindex % nColumns)) * ((n % nColumns) - (BMUindex % nColumns)));
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
                else
                {
                	for (int n = 0; n < nNeuron; n++)
                	{
				        int distance = ComputeDistanceHexGrid(BMUindex / nColumns, BMUindex % nColumns, n / nColumns, n % nColumns);
				        if (distance <= radius)
				        {
				            double neigh = 0;
				            // compute neigh param as requested
				            switch (neighborsType)
				            {
				                case 'g' : neigh = gaussian(distance, radius); break;
				                case 'b' : neigh = bubble(distance, radius); break;
				                case 'm' : neigh = mexican_hat(distance, radius); break;
				            }
				            // update all features of the neuron
				            for (int i = n * nElements, j=0; j < nElements; i++,j++)
				            {
				            	double tmp = neigh * k_Samples[index * nElements + j];
				            	atomicAdd(&k_Matrix_num[i], tmp); 
				            }
				            atomicAdd(&k_Matrix_denum[n], neigh);
				        }
                	}
                }
            }
        }
	}
}

__global__ void batch_update(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, int nElements, int nNeuron)
{
	// getting the index of the thread
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		// store the influence of the divisor
		double tmp = k_Matrix_denum[index];
		if (tmp != 0)
		{
			// save to the matrix the new neuron
			for (int i = 0; i < nElements; i++)
			{
				k_Matrix[index * nElements + i] = k_Matrix_num[index * nElements + i] / tmp;
			}
		}
	}
}
