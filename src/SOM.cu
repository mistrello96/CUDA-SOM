#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <random>
#include <float.h>

#include "utility_functions.cpp"
#include "distance_kernels.cu"
#include "cmdline.h"

#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(-13);															\
} }

int main(int argc, char **argv)
{
	// reading passed params
	gengetopt_args_info ai;
    if (cmdline_parser (argc, argv, &ai) != 0)
    {
        exit(1);
    }
	// INIZIALIZING VARIABLES WITH DEFAULT VALUES
	// path of the input file
	std::string filePath = ai.inputfile_arg;
	// verbose flag
    bool verbose = ai.verbose_flag;
    // advanced debug flag
    bool debug = ai.debug_flag;
    // save SOM to file
    bool print = ai.save_flag;
    // number of rows in the martix
    int nRows = ai.nRows_arg;
    // number of column in the martix
    int nColumns = ai.nColumns_arg;
    // initial learning rate
    double ilr = ai.initial_learning_rate_arg;
    // final learning rate
    double flr = ai.final_learning_rate_arg;
    // max number of iteration
    int maxnIter = ai.iteration_arg;
    // accuracy threshold
    double accuracyTreshold = ai.accuracy_arg;
    // Initial radius of the update
    double initialRadius = ai.radius_arg;
    // type of distance used
    char distanceType = ai.distance_arg[0];
    // enable the normalization of the distance fuction
    bool normalizeFlag = ai.normalize_flag;
    // type of neighbors function used
    char neighborsType = ai.neighbors_arg[0];
    // type of initialization
    char initializationType = ai.initialization_arg[0];
    // type of lactice used
    char lacticeType = ai.lactice_arg[0];
    // exponential decay for radius and lr
    char exponential = ai.exponential_arg[0];
    // dataset presentation methon
    bool randomizeDataset = ai.randomize_flag;
    // counter for times of Samples vector is presented to the SOM
    int nIter = 0;
    // declaration of some usefull variables
    double min_neuronValue, max_neuronValue;
    // number of lines in the input file
    int nSamples;
    // total number of neurons in the SOM
    int nNeurons;
    // total length of the matrix vector
    int totalLength;
    // number of features per read
    int nElements;
    // number of blocks that needs to be launched
    int nblocks;
    // actual learning rate
    double lr;
    // actual radius
    double radius;
    // actual accuracy
	double accuracy;

    // READ THE INPUT FILE
    // vector of samples to be analized from the SOM
    std::vector <double> Samples;
    // retrive the number of features readed from the file
    nElements = readSamplesfromFile(Samples, filePath);

    // EXTRACTING THE MIN/MAX FROM SAMPLES(only used for random initialization)
    if (initializationType == 'r')
    {
	    // creating the thrust vector
	    thrust::device_vector<double> t_Samples(Samples);
	    // extract the minimum
	    thrust::device_vector<double>::iterator it = thrust::min_element(t_Samples.begin(), t_Samples.end());
	    min_neuronValue = *it;
	    // extract maximum
	    thrust::device_vector<double>::iterator it2 = thrust::max_element(t_Samples.begin(), t_Samples.end());
	    max_neuronValue = *it2;
	}

    // COMPUTE VALUES FOR THE SOM INITIALIZATION
    // retrive the number of samples
    nSamples = Samples.size() / nElements;

    // estimate the neurons number if not given
    if (nRows ==0 | nColumns == 0)
    {
    	int tmp = 5*(pow(nSamples, 0.54321));
    	nRows = sqrt(tmp);
    	nColumns = sqrt(tmp);
    }

    // estimate the radius if not given (covering 2/3 of the matrix)
    if (initialRadius == 0)
    	initialRadius = 1 + (max(nRows, nColumns)/2) * 2 / 3;

    // total number of neurons in the SOM
    nNeurons = nRows * nColumns;
    // total length of the serialized matrix
    totalLength = nRows * nColumns * nElements;
    
    // number of block used in the computation
    nblocks = (nNeurons / getnThreads()) + 1;

    // CHECKING COMPUTABILITY ON CUDA
    if (nblocks >= 65535)
    {
    	std::cout << "Too many bocks generated, cannot run a kernel with so many blocks. Try to reduce the number of neurons" << std::endl;
    	exit(-1);
    }

    // CHECK AVAILABLE MEMORY
    if (sizeof(double) * nNeurons * nElements >= checkFreeGpuMem())
    {
	    	std::cout << "Not enougth memory on the GPU, try to reduce neurons' number" << std::endl;
	    	exit(-1);
	}

    // ALLOCATION OF THE STRUCTURES
    // host SOM
    double *h_Matrix = (double *)malloc(sizeof(double) * totalLength);
    // host sample array
    double *h_ActualSample = (double *)malloc(sizeof(double) * nElements);
    // host distance array, used to find BMU
    double *h_Distance = (double *) malloc(sizeof(double) * nNeurons);
    // device SOM
    double *d_Matrix;
    // device sample array
    double *d_ActualSample;
    // device distance array, 
    double *d_Distance;
    // device malloc
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_Matrix, sizeof(double) * totalLength));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_ActualSample, sizeof(double) * nElements));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Distance, sizeof(double) * nNeurons));

    // SOM INIZIALIZATION
    // generating random seed
    std::random_device rd;
    std::mt19937 e2(rd());
    if (initializationType == 'r')
    {
    	std::uniform_real_distribution<> dist(min_neuronValue, max_neuronValue);
	    for(int i = 0; i < totalLength; i++)
	    {
	    	h_Matrix[i] = dist(e2); 
	    }
    }
    else if (initializationType == 'c')
    {
    	std::uniform_int_distribution<> dist(0, nSamples);
	    for (int i = 0; i < nNeurons; i++)
	    {
	        int r = dist(e2);
	        for (int k = i * nElements, j = 0; j < nElements; k++, j++)
	        {
	             h_Matrix[k] = Samples[r*nElements + j];
	        }
	    }
	}
	else
    {
		// TODO PCA
	}

    if (debug | print)
        saveSOMtoFile("initialSOM.out", h_Matrix, nRows, nColumns, nElements);
    
	// inizializing actual values
    lr = ilr;
    radius = initialRadius;
	accuracy = DBL_MAX;

    // debug print
    if(verbose | debug)
    {
        std::cout << "Running the program with " << nRows  << " rows, " << nColumns << " columns, " << nNeurons << " neurons, " << nElements << " features fot each read, " << ilr << " initial learning rate, " << flr << " final learning rate, " << accuracyTreshold<< " required accuracyTreshold, " << radius << " initial radius, ";
        std::cout << maxnIter << " max total iteration, " << distanceType << " distance type, " << normalizeFlag << " normalized, " << neighborsType << " neighbors function, ";
        std::cout << initializationType << " initialization teqnique, " << lacticeType << " lactice, " << exponential << " type of decay, " << randomizeDataset << " randomized input, " << nSamples << " sample in the input file" << nblocks << "blocks will be launched on the GPU" << std::endl;
    
    }

    // initializing indexes to shuffle the Samples vector
    int randIndexes[nSamples];
    for (int i = 0; i < nSamples; i++)
    {
    	randIndexes[i] = i;
    }

    // thrust vector used to store the BMU distances of each iteration
    thrust::device_vector<double> d_DistanceHistory;

    while((accuracy >= accuracyTreshold) && (lr >= flr) && (nIter < maxnIter))
    {
    	// randomize indexes of samples
    	if(randomizeDataset)
    		std::random_shuffle(&randIndexes[0], &randIndexes[nSamples-1]);

        if (debug | verbose)
        {
            std::cout << "Learning rate of this iteration is " << lr << std::endl;
            std::cout << "Radius of this iteration is " << radius << std::endl;
        }

        // ITERATE ON EACH SAMPLE TO FIND BMU
	    for(int s=0; s < nSamples ; s++)
        {

		    // copy the s sample in the actual sample vector
		    for(int i = randIndexes[s]*nElements, j = 0; i < randIndexes[s]*nElements+nElements; i++, j++)
            {
		    	h_ActualSample[j] = Samples[i];
		    } 

			// copy from host to device matrix, actual sample and distance
			CUDA_CHECK_RETURN(cudaMemcpy(d_Matrix, h_Matrix, sizeof(double) * totalLength, cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy(d_ActualSample, h_ActualSample, sizeof(double) * nElements, cudaMemcpyHostToDevice));
			
		    // parallel search launch
		    if (normalizeFlag)
            {
		    	switch(distanceType)
                {
		    		case 'e' : compute_distance_euclidean_normalized<<<nblocks, 1024>>>(d_Matrix, d_ActualSample, d_Distance, nNeurons, nElements); break;
		    		case 's' : compute_distance_sum_squares_normalized<<<nblocks, 1024>>>(d_Matrix, d_ActualSample, d_Distance, nNeurons, nElements); break;
		    		case 'm' : compute_distance_manhattan_normalized<<<nblocks, 1024>>>(d_Matrix, d_ActualSample, d_Distance, nNeurons, nElements); break;
		    		case 't' : compute_distance_tanimoto<<<nblocks, 1024>>>(d_Matrix, d_ActualSample, d_Distance, nNeurons, nElements); break;
		    	}
		    }
            else
            {
		    	switch(distanceType)
                {
		    		case 'e' : compute_distance_euclidean<<<nblocks, 1024>>>(d_Matrix, d_ActualSample, d_Distance, nNeurons, nElements); break;
		    		case 's' : compute_distance_sum_squares<<<nblocks, 1024>>>(d_Matrix, d_ActualSample, d_Distance, nNeurons, nElements); break;
		    		case 'm' : compute_distance_manhattan<<<nblocks, 1024>>>(d_Matrix, d_ActualSample, d_Distance, nNeurons, nElements); break;
				    case 't' : compute_distance_tanimoto<<<nblocks, 1024>>>(d_Matrix, d_ActualSample, d_Distance, nNeurons, nElements); break;
		    	}
		    }

			//wait for all block to complete the computation
		    cudaDeviceSynchronize();

		    // CHECK AVAILABLE MEMORY
	    	if (sizeof(double) * nNeurons * nElements >= checkFreeGpuMem())
            {
		    	std::cout << "Out of memory, try to reduce the neurons number" << std::endl;
		    	exit(-1);
			}
				
			// create thrust vector to find BMU  in parallel
			thrust::device_vector<double> d_vec_Distance(d_Distance, d_Distance + nNeurons);
			// extract the first matching BMU
			thrust::device_vector<double>::iterator iter = thrust::min_element(d_vec_Distance.begin(), d_vec_Distance.end());
			// extract index and value of BMU
			unsigned int BMU_index = iter - d_vec_Distance.begin();
            unsigned int BMU_x = BMU_index / nColumns;
            unsigned int BMU_y = BMU_index % nColumns;
			double BMU_distance = *iter;

            double tmp = 0;
            double dist = 0;
            for(int u = 0; u < nElements; u++)
            {
                tmp = h_Matrix[BMU_index*nElements + u] - h_ActualSample[u];
                dist += tmp * tmp; 
            }


			// adding the found value in the distance history array
			d_DistanceHistory.push_back(dist);

			// debug print
		    if(debug)
			   std::cout << "The minimum distance is " << dist << " at position " << BMU_index << std::endl;

			// UPDATE THE NEIGHBORS
			// if radius is 0, update only BMU 
	        if (radius == 0)
	        {
	        	for (int i = BMU_index * nElements, j = 0; j < nElements; i++, j++)
                {
	        		h_Matrix[i] = h_Matrix[i] + lr * (h_ActualSample[j] - h_Matrix[i]);
	        	}
	        }
	        // possible to transfer on the gpu?
            // update also the neighbors
	        else
	        {
	            for (int i = 0; i < nNeurons; i++){
	                int x = i / nColumns;
	                int y = i % nColumns;
                    int distance = 0;
                    if (lacticeType == 's')
	                   distance = sqrt((x - BMU_x) * (x - BMU_x) + (y - BMU_y) * (y - BMU_y));
                    else
                        distance = ComputeDistanceHexGrid(BMU_x, BMU_y, x, y);
                    
                    // update only if...
	                if (distance <= radius)
                    {
                        double neigh = 0.0;
                        switch (neighborsType)
                        {
                            case 'g' : neigh = gaussian(distance, radius); break;
                            case 'b' : neigh = bubble(distance, radius); break;
                            case 'm' : neigh = mexican_hat(distance, radius); break;
                        } 

	                    for (int k = i * nElements, j = 0; j < nElements; k++, j++)
                        {
	                        h_Matrix[k] = h_Matrix[k] + neigh * lr * (h_ActualSample[j] - h_Matrix[k]);
	                    }
	                }
	            }
        	}	         
        }

        // END OF SAMPLES ITERATION. UPDATING VALUES
        // updating accuracy
        //accuracy = thrust::reduce(d_DistanceHistory.begin(),d_DistanceHistory.end())/ ((double)nSamples);
        //d_DistanceHistory.clear();
        accuracy = thrust::reduce(d_DistanceHistory.begin(), d_DistanceHistory.end());
        accuracy = sqrt(accuracy/nElements)/nSamples;
        d_DistanceHistory.clear();

        if (verbose | debug)
        {
            std::cout << "Mean distance of this iteration is " << accuracy << std::endl;
        }
        
		// updating the counter iteration
		nIter ++;

        // updating radius and learning rate
        radius = (int) (initialRadius - (initialRadius) * ((double)nIter/maxnIter));
        if (exponential== 'r' | exponential == 'b')
        	radius = (int) (initialRadius * exp(-(double)nIter/(sqrt(maxnIter)))) ;

        lr = ilr - (ilr - flr) * ((double)nIter/maxnIter);
        if (exponential== 'l' | exponential == 'b')
        	lr = ilr * exp(- (double)nIter/sqrt(maxnIter)) + flr;
    }

    if (debug | print)
    {
        saveSOMtoFile("outputSOM.out", h_Matrix, nRows, nColumns, nElements);
    }

	//freeing all allocated memory
    cudaFree(d_Matrix);
    cudaFree(d_ActualSample);
    cudaFree(d_Distance);
    free(h_Matrix);
    free(h_Distance);
    free(h_ActualSample);
}