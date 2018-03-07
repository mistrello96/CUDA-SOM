//#include <stdio.h>
//#include <iostream>
//#include <cuda_runtime.h>
//#include <string>
//#include <cmath>
#include <ctime>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <unistd.h>
#include <fstream>
#include <sstream>

#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(-13);															\
} }


__global__ void compute_distance(float* k_matrix, float* k_ActualSample, float* k_distance, int nNeuron, int nElements){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		int matrixindex = index * nElements;
		float tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + powf(k_matrix[matrixindex+i] - k_ActualSample[i], 2.0);
		}

		k_distance[index] = sqrtf(tmp);
	}
}


float checkFreeGpuMem(){
	float free_m;
	size_t free_t,total_t;
	cudaMemGetInfo(&free_t,&total_t);
	free_m =(uint)free_t;
	return free_m;
}


// returns the number of features per line
int readSamplesfromFile(std::vector<float>& samples, std::string filePath){
	int counter = 0;
	std::string line;
	std::ifstream file (filePath.c_str());
	if (file.is_open()) {
		while (std::getline (file, line)) {
			std::istringstream iss(line);
    		std::string element;
    		int tmp = 0;
    		while(std::getline(iss, element, '\t')){
    			tmp ++;
				samples.push_back(strtof((element).c_str(),0));
			}
			counter = tmp;
		}
		file.close();
		return counter;
	}
	else{
		std::cout << "Unable to open file";
		exit(-1);
	}
}


int main(int argc, char **argv)
{
	// INIZIALIZING VARIABLES WITH DEFAULT VALUES
	// path of the input file
	std::string filePath = "./";
	//debuf flag
    bool debug = false;
    // number of rows in the martix
    int nRows = 50;
    // number of column in the martix
    int nColumns = 50;

    // COMMAND LINE PARSING
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

    // READ THE INPUT FILE
    // vector of samples to be analized from the SOM
    std::vector <float> Samples;
    // retrive the number of features readed from the file
    int nElements = readSamplesfromFile(Samples, filePath);

    // COMPUTE USEFULL VALUES
    // total number of neurons in the SOM
    int nNeurons = nRows * nColumns;
    // total length of the serialized matrix
    int totalLength = nRows * nColumns * nElements;
    // number of block used in the computation
    int nblocks = (nNeurons / 1024) + 1;
    // checking the computability on CUDA
    if (nblocks >= 65535){
    	std::cout << "Too many bocks generated, cannot run a kernel with so many blocks. Try to reduce the number of neurons" << std::endl;
    	exit(-1);
    }
    // retrive the number of samples
    int nSamples = Samples.size() / nElements;

    // CHECK AVAILABLE MEMORY
    if (sizeof(float) * nNeurons * nElements >= checkFreeGpuMem()){
	    	std::cout << "Not enougth memory on the GPU, try to reduce neurons' number" << std::endl;
	    	exit(-1);
	}

    // debug print
    if(debug){
        std::cout << "Running the program with " << nRows  << " rows, " << nColumns << " columns, " << nNeurons << " neurons, " << nElements << " features." << std::endl;
    }

    // ALLOCATION OF THE STRUCTURES
    // host SOM
    float *h_Matrix = (float *)malloc(sizeof(float) * totalLength);
    // host sample array
    float *h_ActualSample = (float *)malloc(sizeof(float) * nElements);
    // host distance array, used to find BMU
    float *h_Distance = (float *) malloc(sizeof(float) * nNeurons);
    // device SOM
    float *d_Matrix;
    // device sample array
    float *d_Sample;
    // device distance array, 
    float *d_Distance;
    // device malloc
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_Matrix, sizeof(float) * totalLength));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Sample, sizeof(float) * nElements));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Distance, sizeof(float) * nNeurons));

    // SOM INIZIALIZATION
    // generating random seed
    srand(time(NULL));
    // random values SOM initialization
    for(int i = 0; i < totalLength; i++){
    	h_Matrix[i] = rand() % 100;
    }

    // ITERATE ON EACH SAMPLE TO FIND BMU
    for(int s=0; s < nSamples ; s++){

	    // distance array inizialization
	    for(int i = 0; i < nNeurons; i++){
	    	h_Distance[i] = 0;
	    }

	    // copy the s sample in the actual sample vector
	    for(int i = s*nElements, j = 0; i < s*nElements+nElements; i++, j++){
	    	h_ActualSample[j] = Samples[i];
	    }; 

		// copy from host to device matrix, actual sample and distance
		CUDA_CHECK_RETURN(cudaMemcpy(d_Matrix, h_Matrix, sizeof(float) * totalLength, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_Sample, h_ActualSample, sizeof(float) * nElements, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_Distance, h_Distance, sizeof(float) * nNeurons, cudaMemcpyHostToDevice));	
		
	    // parallel search launch
	    compute_distance<<<nblocks,1024>>>(d_Matrix, d_Sample, d_Distance, nNeurons, nElements);

		//wait for all block to complete the computation
	    cudaDeviceSynchronize();

	    /*
	    cudaMemcpy(h_Distance, d_Distance, sizeof(float) * nNeurons, cudaMemcpyDeviceToHost);
	    
	    for(int i = 0; i < nNeurons; i++){
	        std::cout << h_Distance[i] << std::endl;
	    }
	    */
	    /*
	    if (sizeof(float) * nNeurons >= checkFreeGpuMem()){
	    	std::cout << "Out of memory" << sizeof(float) * nNeurons << "    " << checkFreeGpuMem() << std::endl;
	    	exit(-1);
	    }
	    */

	    // CHECK AVAILABLE MEMORY
    	if (sizeof(float) * nNeurons * nElements >= checkFreeGpuMem()){
	    	std::cout << "Out of memory, try to reduce the neurons number" << std::endl;
	    	exit(-1);
		}
			
		// create thrust vector to find BMU  in parallel
		thrust::device_vector<float> d_vec_Distance(d_Distance, d_Distance + nNeurons);
		// extract the first matching BMU
		thrust::device_vector<float>::iterator iter = thrust::min_element(d_vec_Distance.begin(), d_vec_Distance.end());
		// extract index and value of BMU
		unsigned int BMU_index = iter - d_vec_Distance.begin();
		float BMU_value = *iter;

		// debug print
	    if(debug)
		   std::cout << "The minimum value is " << BMU_value << " at position " << BMU_index << std::endl;
		//TODO: update BMU and neighbors
	}

	//freeing all allocated memory
    cudaFree(d_Matrix);
    cudaFree(d_Sample);
    cudaFree(d_Distance);
    free(h_Matrix);
    free(h_Distance);
    free(h_ActualSample);

}

