#ifndef BATCH_DISTANCE_KERNELS_H
#define BATCH_DISTANCE_KERNELS_H

// kernel used to compute euclidean distance between each neuron and the selected sample
__global__ void batch_compute_distance_euclidean(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, double* k_Samples, double* k_Distance, int * k_BMU, int nSamples, int nNeuron, int nElements, bool normalizedistance, char neighborsType, int nRows, int nColumns, bool toroidal, int radius, char lattice);

// kernel used to compute sum of squares distance between each neuron and the selected sample
__global__ void batch_compute_distance_sum_squares(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, double* k_Samples, double* k_Distance, int * k_BMU, int nSamples, int nNeuron, int nElements, bool normalizedistance, char neighborsType, int nRows, int nColumns, bool toroidal, int radius, char lattice);

// kernel used to compute manhattan distance between each neuron and the selected sample
__global__ void batch_compute_distance_manhattan(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, double* k_Samples, double* k_Distance, int * k_BMU, int nSamples, int nNeuron, int nElements, bool normalizedistance, char neighborsType, int nRows, int nColumns, bool toroidal, int radius, char lattice);

// kernel used to compute tanimoto distance between each neuron and the selected sample
__global__ void batch_compute_distance_tanimoto(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, double* k_Samples, double* k_Distance, int * k_BMU, int nSamples, int nNeuron, int nElements, bool normalizedistance, char neighborsType, int nRows, int nColumns, bool toroidal, int radius, char lattice);

__global__ void batch_update(double* k_Matrix, double* k_Matrix_num, double* k_Matrix_denum, int nElements, int nNeuron);

#endif