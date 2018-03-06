#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

__global__ void compute_distance(float* k_matrix, int nNeuron, float* k_sample, float* k_distance, int sampleLength)
{
	int index = (threadIdx.x + blockIdx.x * blockDim.x);
	if (index < nNeuron)
	{
		int matrixindex = index * 14;
		float tmp = 0;
		for(int i = 0; i < sampleLength; i++)
		{
			tmp = tmp + abs(k_matrix[matrixindex+i] - k_sample[i]);
		}

		k_distance[index] = k_distance[index] + tmp;
	}
}

void readSamplesfromFile(std::vector<float>& samples, std::string filePath){
	std::string line;
	std::ifstream file (filePath.c_str());
	if (file.is_open()) {
		while (std::getline (file, line) ) {
			std::istringstream iss(line);
    		std::string element;
    		while(std::getline(iss, element, '\t'))
				samples.push_back(strtof((element).c_str(),0));
		}
		file.close();
	}
	else{
		std::cout << "Unable to open file";
		exit(-1);
	}
}


int main(int argc, char **argv)
{
	std::string filePath = "./";
    bool debug = false;
	// number of features in each neuron
    int nElements = 0;
    // number of rows in the martix
    int nRows = 1000;
    // number of column in the martix
    int nColumns = 1000;

    //command line parsing
    int c;
    while ((c = getopt (argc, argv, "i:n:x:y:hv")) != -1)
    switch (c)
	{
        case 'i':
            filePath = optarg;
            break;
        case 'n':
            if (int (sqrt(atoi(optarg))) * int (sqrt(atoi(optarg))) != atoi(optarg)){
                std::cout << "The -x option only support square matrix. To create a generic matrix, use -x and -y parameters" << std::endl;
                return(-1);
            }
            nRows = sqrt(atoi(optarg));
            nColumns = sqrt(atoi(optarg));
            break;
        case 'x':
            nRows = atoi(optarg);
            break;
        case 'y':
            nColumns = atoi(optarg);
            break;
        case 'v':
            debug = true;
            break;
        case 'h':
            std::cout << "-i allows to provide the PATH of an input file" << std::endl;
            std::cout << "-n allows to provide the neurons's number in the SOM(only for square matrix)" << std::endl;
            std::cout << "-x allows to provide the number of rows in the neuron's matrix" << std::endl;
            std::cout << "-y allows to provide the numbers of columns in the neuron's matrix" << std::endl;
            std::cout << "-d enables debug prints" << std::endl;
            std::cout << "-h shows help menu of the tool" << std::endl;
            return 0;
    }

    // total number of neurons in the SOM
    int nNeurons = nRows * nColumns;
    // total length of the serialized matrix
    int totalLength = nRows * nColumns * nElements;
    //vector of samples to be analized from the SOM
    std::vector <float> Sample;
    for (int i = 0; i < Sample.size(); i++){
    	std::cout << Sample[i] << std::endl;
    }
    readSamplesfromFile(Sample, filePath);

    // host SOM
    float *h_Matrix = (float *)malloc(sizeof(float) * totalLength);
    // host sample array
    float *h_ActualSample = (float *)malloc(sizeof(float) * nElements);
    // host distance array, used to find BMU
    float *h_Distance = (float *) malloc(sizeof(float) * nNeurons);

    if(debug){
        std::cout << "Running the program with " << nRows  << " rows, " << nColumns << " columns, " << nNeurons << " neurons, " << nElements << " features." << std::endl;
    }

    //random SOM initialization
    for(int i = 0; i < totalLength; i++){
    	h_Matrix[i] = i;
    }
    // distance array inizialization
    for(int i = 0; i < nNeurons; i++){
    	h_Distance[i] = 0;
    }

    //random sample inizialization, used for TEST
    for(int i = 0; i < nElements; i++){
    	h_ActualSample[i] = i+1;
    }

    // device SOM
    float *d_Matrix;
    // device sample array
    float *d_Sample;
    // device distance array, 
    float *d_Distance;

    //device malloc
    cudaMalloc((void **)&d_Matrix, sizeof(float) * totalLength);
    cudaMalloc((void**)&d_Sample, sizeof(float) * nElements);
    cudaMalloc((void**)&d_Distance, sizeof(float) * nNeurons);

	//copy from host to device matrix, sample and distance
	cudaMemcpy(d_Matrix, h_Matrix, sizeof(float) * totalLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sample, h_ActualSample, sizeof(float) * nElements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Distance, h_Distance, sizeof(float) * nNeurons, cudaMemcpyHostToDevice);	
	
    //peparing param to launch kernel
    int nblocks = (nNeurons / 1024) + 1; 
    compute_distance<<<nblocks,1024>>>(d_Matrix, nNeurons, d_Sample, d_Distance, nElements);

	//wait for all block to be completed
    cudaDeviceSynchronize();

    /*
    cudaMemcpy(h_Distance, d_Distance, sizeof(float) * nNeurons, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < nNeurons; i++){
        std::cout << h_Distance[i] << std::endl;
    }
    */

	//create thrust vector to find BMU
	thrust::device_vector<float> d_vec_Distance(d_Distance, d_Distance + nNeurons);
	//extract the first element
	thrust::device_vector<float>::iterator iter = thrust::min_element(d_vec_Distance.begin(), d_vec_Distance.end());
	// find index of BMU
	unsigned int BMU_index = iter - d_vec_Distance.begin();
	float BMU_value = *iter;

    if(debug)
	   std::cout << "The minimum value is " << BMU_value << " at position " << BMU_index << std::endl;
	//TODO: update BMU and neighbors
 
    cudaFree(d_Matrix);
    free(h_Matrix);

}

