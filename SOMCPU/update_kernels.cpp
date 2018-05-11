// kernel to update SOM after the BMU has been found. Called only if radius of the update is 0, so only BMU will be updated
void update_BMU(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex)
{
	// update all features of the BMU
    for (int i = BMUIndex * nElements, j=0; j < nElements; i++, j++)
    {
        k_Matrix[i] = k_Matrix[i] + lr * (k_Samples[samplesIndex + j] - k_Matrix[i]); 
    }
}

// kernel to update SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
void update_SOM(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nColumns, int radius, int nNeuron, char neighborsType)
{
    for(int index = 0; index < nNeuron; index++)
    {
        // compute distance if lattice is square
        int distance = sqrtf(((index / nColumns) - (BMUIndex / nColumns)) * ((index / nColumns) - (BMUIndex / nColumns)) + ((index % nColumns) - (BMUIndex % nColumns)) * ((index % nColumns) - (BMUIndex % nColumns)));
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
            for (int i = index * nElements, j=0; j < nElements; i++,j++)
            {
                k_Matrix[i] = k_Matrix[i] + neigh * lr * (k_Samples[samplesIndex + j] - k_Matrix[i]);
            }
        }
    }
}

// kernel to update SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
void update_SOM_toroidal(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nRows, int nColumns, int radius, int nNeuron, char neighborsType)
{
    for(int index = 0; index < nNeuron; index++)
    {
        // call function to compute distance in a toroidal square map
        int distance = ComputeDistanceToroidal(index / nColumns, index % nColumns, BMUIndex / nColumns, BMUIndex % nColumns, nRows, nColumns);
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
            for (int i = index * nElements, j=0; j < nElements; i++,j++)
            {
                k_Matrix[i] = k_Matrix[i] + neigh * lr * (k_Samples[samplesIndex + j] - k_Matrix[i]);
            }
        }
    }
}

// kernel to update a exagonal SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
void update_SOM_exagonal(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nColumns, int radius, int nNeuron, char neighborsType)
{
    for(int index = 0; index < nNeuron; index++)
    {
        // call function to compute distance in a exagonal map
        int distance = ComputeDistanceHexGrid(BMUIndex / nColumns, BMUIndex % nColumns, index / nColumns, index % nColumns);
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
            for (int i = index * nElements, j=0; j < nElements; i++,j++)
            {
                k_Matrix[i] = k_Matrix[i] + neigh * lr * (k_Samples[samplesIndex + j] - k_Matrix[i]);
            }
        }
    }
}

// kernel to update a exagonal toroidal SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
void update_SOM_exagonal_toroidal(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nRows, int nColumns, int radius, int nNeuron, char neighborsType)
{
    for(int index = 0; index < nNeuron; index++)
    {
        // call function to compute distance in a toroidal exagonal map
        int distance = ComputeDistanceHexGridToroidal(BMUIndex / nColumns, BMUIndex % nColumns, index / nColumns, index % nColumns, nRows, nColumns);
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
            for (int i = index * nElements, j=0; j < nElements; i++,j++)
            {
                k_Matrix[i] = k_Matrix[i] + neigh * lr * (k_Samples[samplesIndex + j] - k_Matrix[i]);
            }
        }
    }
}