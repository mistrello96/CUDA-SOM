// kernel to update SOM after the BMU has been found. Called only if radius of the update is 0, so only BMU will be updated
__global__ void update_BMU(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, char neighborsType)
{
	// update all features of the BMU
    for (int i = BMUIndex * nElements, j=0; j < nElements; i++, j++)
    {
        k_Matrix[i] = k_Matrix[i] + lr * (k_Samples[samplesIndex + j] - k_Matrix[i]); 
    }
}

// kernel to update SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
__global__ void update_SOM(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nColumns, int radius, int nNeuron, char neighborsType)
{
	// compute neuron's index
    int threadindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadindex < nNeuron){
        //int x = threadindex / nColumns;
        //int y = threadindex % nColumns;
        //int BMU_x = BMUIndex / nColumns;
        //int BMU_y = (BMUIndex % nColumns);
        // compute distance if lattice is square
        int distance = sqrtf(((threadindex / nColumns) - (BMUIndex / nColumns)) * ((threadindex / nColumns) - (BMUIndex / nColumns)) + ((threadindex % nColumns) - (BMUIndex % nColumns)) * ((threadindex % nColumns) - (BMUIndex % nColumns)));
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
            for (int i = threadindex * nElements, j=0; j < nElements; i++,j++)
            {
                k_Matrix[i] = k_Matrix[i] + neigh * lr * (k_Samples[samplesIndex + j] - k_Matrix[i]);
            }
        }
    }
}

// kernel to update SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
__global__ void update_SOM_toroidal(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nRows, int nColumns, int radius, int nNeuron, char neighborsType)
{
	// compute neuron's index
    int threadindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadindex < nNeuron){
        int X = abs((threadindex / nColumns) - (BMUIndex / nColumns));
        int Y = abs((threadindex % nColumns) - (BMUIndex % nColumns));
        if (X > abs(threadindex / nColumns)+nColumns-(BMUIndex / nColumns))
       		X =	abs(threadindex / nColumns)+nColumns-(BMUIndex / nColumns);
       	if (Y > abs(threadindex % nColumns)+nRows-(BMUIndex % nColumns))
       		Y = abs(threadindex % nColumns)+nRows-(BMUIndex % nColumns);
        // compute distance if lattice is square
        int distance = sqrtf((X * X) + (Y * Y));
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
            for (int i = threadindex * nElements, j=0; j < nElements; i++,j++)
            {
                k_Matrix[i] = k_Matrix[i] + neigh * lr * (k_Samples[samplesIndex + j] - k_Matrix[i]);
            }
        }
    }
}

__global__ void update_SOM_exagonal(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nColumns, int radius, int nNeuron, char neighborsType)
{
	// compute neuron's index
    int threadindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadindex < nNeuron){
        //int x = threadindex / nColumns;
        //int y = threadindex % nColumns;
        //int BMU_x = BMUIndex / nColumns;
        //int BMU_y = (BMUIndex % nColumns);
        // compute distance if lattice is exagonal
        int distance = ComputeDistanceHexGrid(BMUIndex / nColumns, BMUIndex % nColumns, threadindex / nColumns, threadindex % nColumns);
        if (distance <= radius)
        {
            double neigh =0;
            // compute neigh param as requested
            switch (neighborsType)
            {
                case 'g' : neigh = gaussian(distance, radius); break;
                case 'b' : neigh = bubble(distance, radius); break;
                case 'm' : neigh = mexican_hat(distance, radius); break;
            }
            // update all features of the neuron
            for (int i = threadindex * nElements, j=0; j < nElements; i++,j++)
            {
                k_Matrix[i] = k_Matrix[i] + neigh * lr * (k_Samples[samplesIndex + j] - k_Matrix[i]);
            }
        }
    }
}