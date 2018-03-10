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
#include <float.h>

#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(-13);															\
} }


__global__ void compute_distance(double* k_matrix, double* k_ActualSample, double* k_distance, int nNeuron, int nElements){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < nNeuron)
	{
		int matrixindex = index * nElements;
		double tmp = 0;
		for(int i = 0; i < nElements; i++)
		{
			tmp = tmp + powf(k_matrix[matrixindex+i] - k_ActualSample[i], 2.0);
		}

		k_distance[index] = sqrtf(tmp);
		//printf("%f\n", k_distance[index]);
	}
}


double checkFreeGpuMem(){
	double free_m;
	size_t free_t,total_t;
	cudaMemGetInfo(&free_t,&total_t);
	free_m =(uint)free_t;
	return free_m;
}


// returns the number of features per line
int readSamplesfromFile(std::vector<double>& samples, std::string filePath){
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
	// debuf flag
    bool debug = false;
    // number of rows in the martix
    int nRows = 0;
    // number of column in the martix
    int nColumns = 0;
    // initial learning rate
    double ilr = 0;
    // final learning rate
    double flr = 0;
    // lambda of the gaussian
    double lambda = 0;
    // max number of iteration
    int maxnIter = 0;
    // accuracy threshold
    double requiredAccuracy = 0;
    // learning iteration counter
    int nIter = 0;

    // COMMAND LINE PARSING
    int c;
    while ((c = getopt (argc, argv, "i:x:y:hvs:f:m:a:l:")) != -1)
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
        case 's':
            ilr = strtof(optarg,0);
            break;
        case 'f':
            flr = strtof(optarg,0);
            break;
        case 'a':
            requiredAccuracy = strtof(optarg,0);
            break;
        case 'l':
            lambda = strtof(optarg,0);
            break;
        case 'm':
            maxnIter = atoi(optarg);
            break;
        case 'v':
            debug = true;
            break;
        case 'h':
            std::cout << "-i allows to provide the PATH of an input file. If not specified, ./ is assumed" << std::endl;
            std::cout << "-x allows to provide the number of rows in the neuron's matrix. REQUESTED" << std::endl;
            std::cout << "-y allows to provide the numbers of columns in the neuron's matrix. REQUESTED" << std::endl;
            std::cout << "-s initial learning rate" << std::endl;
            std::cout << "-f final learning rate" << std::endl;
            std::cout << "-l lambda of the gaussian" << std::endl;
            std::cout << "-a accuracy threshold" << std::endl;
            std::cout << "-m maximum number of iteration before stopping the learning process" << std::endl;
            std::cout << "-v enables debug prints" << std::endl;
            std::cout << "-h shows help menu of the tool" << std::endl;
            return 0;
    }

    // checking the required params
    if (nRows == 0 | nColumns == 0 | ilr == 0 | maxnIter == 0){
        std::cout << "Required params are missing, program will abort" << std::endl;
        exit(-1);        
    }

    // READ THE INPUT FILE
    // vector of samples to be analized from the SOM
    std::vector <double> Samples;
    // retrive the number of features readed from the file
    int nElements = readSamplesfromFile(Samples, filePath);

    // EXTRACTING THE MIN/MAX FROM SAMPLES
    // creating the thrust vector
    thrust::device_vector<double> t_Samples(Samples);
    // extract the minimum
    thrust::device_vector<double>::iterator it = thrust::min_element(t_Samples.begin(), t_Samples.end());
    double min_neuronValue = *it;
    // extract maximum
    thrust::device_vector<double>::iterator it2 = thrust::max_element(t_Samples.begin(), t_Samples.end());
    double max_neuronValue = *it2;

    // COMPUTE USEFULL VALUES
    // total number of neurons in the SOM
    int nNeurons = nRows * nColumns;
    // total length of the serialized matrix
    int totalLength = nRows * nColumns * nElements;
    // number of block used in the computation
    int nblocks = (nNeurons / 1024) + 1;
    // inizializing the learnig rate
    double lr = ilr;
    // retrive the number of samples
    int nSamples = Samples.size() / nElements;

    // CHECKING COMPUTABILITY ON CUDA
    if (nblocks >= 65535){
    	std::cout << "Too many bocks generated, cannot run a kernel with so many blocks. Try to reduce the number of neurons" << std::endl;
    	exit(-1);
    }

    // CHECK AVAILABLE MEMORY
    if (sizeof(double) * nNeurons * nElements >= checkFreeGpuMem()){
	    	std::cout << "Not enougth memory on the GPU, try to reduce neurons' number" << std::endl;
	    	exit(-1);
	}

    // debug print
    if(debug){
        std::cout << "Running the program with " << nRows  << " rows, " << nColumns << " columns, " << nNeurons << " neurons, " << nElements << " features." << std::endl;
    }

    // ALLOCATION OF THE STRUCTURES
    // host SOM
    double *h_Matrix = (double *)malloc(sizeof(double) * totalLength);
    // host sample array
    double *h_ActualSample = (double *)malloc(sizeof(double) * nElements);
    // host distance array, used to find BMU
    double *h_Distance = (double *) malloc(sizeof(double) * nNeurons);
    // host BMU distance array
    double *h_DistanceHistory = (double *)malloc(sizeof(double) * nSamples);
    // device SOM
    double *d_Matrix;
    // device sample array
    double *d_Sample;
    // device distance array, 
    double *d_Distance;
    // device malloc
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_Matrix, sizeof(double) * totalLength));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Sample, sizeof(double) * nElements));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Distance, sizeof(double) * nNeurons));

    // SOM INIZIALIZATION
    // generating random seed
    srand(time(NULL));
    // random values SOM initialization
    for(int i = 0; i < totalLength; i++){
    	double tmp = rand() / (float) RAND_MAX;
    	tmp = min_neuronValue + tmp * (max_neuronValue - min_neuronValue);
    	h_Matrix[i] = tmp; 
    }

    double accuracy = DBL_MAX;
    while((accuracy > requiredAccuracy) && (lr > flr) && (nIter < maxnIter)){
    	// TODO Randomize sample vector
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
			CUDA_CHECK_RETURN(cudaMemcpy(d_Matrix, h_Matrix, sizeof(double) * totalLength, cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy(d_Sample, h_ActualSample, sizeof(double) * nElements, cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy(d_Distance, h_Distance, sizeof(double) * nNeurons, cudaMemcpyHostToDevice));	
			
		    // parallel search launch
		    compute_distance<<<nblocks,1024>>>(d_Matrix, d_Sample, d_Distance, nNeurons, nElements);

			//wait for all block to complete the computation
		    cudaDeviceSynchronize();

		    // CHECK AVAILABLE MEMORY
	    	if (sizeof(double) * nNeurons * nElements >= checkFreeGpuMem()){
		    	std::cout << "Out of memory, try to reduce the neurons number" << std::endl;
		    	exit(-1);
			}
				
			// create thrust vector to find BMU  in parallel
			thrust::device_vector<double> d_vec_Distance(d_Distance, d_Distance + nNeurons);
			// extract the first matching BMU
			thrust::device_vector<double>::iterator iter = thrust::min_element(d_vec_Distance.begin(), d_vec_Distance.end());
			// extract index and value of BMU
			unsigned int BMU_index = iter - d_vec_Distance.begin();
			double BMU_distance = *iter;
			// adding the found value in the distance history array
			h_DistanceHistory[s] = BMU_distance;

			// debug print
		    if(debug)
			   std::cout << "The minimum distance is " << BMU_distance << " at position " << BMU_index << std::endl;

	        //TODO: update BMU and neighbors
	        
	        for (int i = BMU_index * nElements, j = 0; j < nElements; i++, j++){
	        	h_Matrix[i] = h_Matrix[i] + lr*(h_ActualSample[j] - h_Matrix[i]);
	        }
	         
		}

		if (debug){
			std::cout << "Learn rate of this iteration is " << lr << std::endl;
		}
	
		// updating the counter iteration
		nIter ++;
		// updating the learning rate
		lr = ilr - 0.01*nIter;
		// updating accuracy
		thrust::device_vector<double> d_DistanceHistory(h_DistanceHistory, h_DistanceHistory + nSamples);
		double meansum = thrust::reduce(d_DistanceHistory.begin(), d_DistanceHistory.end());
		accuracy = meansum / ((double)nSamples);

		if (debug){
			std::cout << "Mean distance of this iteration is " << accuracy << std::endl;
		}

	}

	//freeing all allocated memory
    cudaFree(d_Matrix);
    cudaFree(d_Sample);
    cudaFree(d_Distance);
    free(h_Matrix);
    free(h_Distance);
    free(h_ActualSample);

}

