#ifndef DISTANCE_KERNELS_H
#define DISTANCE_KERNELS_H

// kernel used to compute euclidean distance between each neuron and the selected sample
__global__ void compute_distance_euclidean(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements);

// kernel used to compute sum of squares distance between each neuron and the selected sample
__global__ void compute_distance_sum_squares(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements);

// kernel used to compute manhattan distance between each neuron and the selected sample
__global__ void compute_distance_manhattan(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements);

// kernel used to compute tanimoto distance between each neuron and the selected sample
__global__ void compute_distance_tanimoto(double* k_matrix, double* k_Samples, int currentIndex, double* k_distance, int nNeuron, int nElements);

#endif