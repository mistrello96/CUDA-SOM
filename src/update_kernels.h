#ifndef UPDATE_KERNELS
#define UPDATE_KERNELS

// kernel to update SOM after the BMU has been found. Called only if radius of the update is 0, so only BMU will be updated
__global__ void update_BMU(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex);

// kernel to update SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
__global__ void update_SOM(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nColumns, int radius, int nNeuron, char neighborsType);

// kernel to update SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
__global__ void update_SOM_toroidal(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nRows, int nColumns, int radius, int nNeuron, char neighborsType);

// kernel to update a exagonal SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
__global__ void update_SOM_exagonal(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nColumns, int radius, int nNeuron, char neighborsType);

// kernel to update a exagonal toroidal SOM after the BMU has been found. Called when radius is > 0, all the SOM neurons will be updated. 
__global__ void update_SOM_exagonal_toroidal(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nRows, int nColumns, int radius, int nNeuron, char neighborsType);

#endif