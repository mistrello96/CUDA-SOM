#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <vector>
#include <iostream>

// returns the number of features per line. Save reads in samples vector
int readSamplesfromFile(std::vector<double>& samples, std::string filePath);

// same the SOM to a file. First two lines are nRows and Ncolumns. Neurons are \n separated, features are \t separated
void saveSOMtoFile(std::string filePath, double* matrix, int nRows, int nColumns, int nElements);

// gaussian distance between two neurons
__device__
double gaussian(double distance, int radius);
// bubble distance between two neurons
__device__
int bubble(double distance, int radius);

// mexican hat distance between two neurons
__device__
double mexican_hat(double distance, int radius);

// compute distance between two neurons on a square toroidal map
__device__ int ComputeDistanceToroidal(int x1, int y1, int x2, int y2, int nRows, int nColumns);

// compute exagonal distance between two neurons	
 __device__ 
int ComputeDistanceHexGrid(int x1, int y1, int x2, int y2);

// compute distance between two neurons on a exagonal toroidal map
 __device__ 
int ComputeDistanceHexGridToroidal(int x1, int y1, int x2, int y2, int nRows, int nColumns);

// run a benchmark to find out the minimum dimension of the input file to make GPU computation advantageous
void run_benchmark();

#endif