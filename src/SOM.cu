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
#include <chrono>

#include "utility_functions.cpp"
#include "distance_kernels.cu"
#include "update_kernels.cu"
#include "cmdline.h"

int main(int argc, char **argv)
{
	// reading passed params
	gengetopt_args_info ai;
    if (cmdline_parser (argc, argv, &ai) != 0)
    {
        exit(1);
    }
    // checking if benchmark mode is set
    if(ai.benchmark_flag){
        run_benchmark();
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
    // type of lacttice used
    char lattice = ai.lattice_arg[0];
    // exponential decay for radius and lr
    char exponential = ai.exponential_arg[0];
    // dataset presentation methon
    bool randomizeDataset = ai.randomize_flag;
    // normalize the mean distance of each iteration
    bool normalizedistance = ai.normalizedistance_flag;
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
	// index of the Samples picked for the iteration
    int currentIndex;
    // BMU distance
    double BMU_distance;
    // BMU index
    unsigned int BMU_index;
    //checking the requested params
    if(ilr == -1 || maxnIter == -1)
    {
        std::cout << "./a.out: '--initial_learning_rate' ('-s') option required" << std::endl;
        std::cout << "./a.out: '--iteration' ('-n') option required " << std::endl;
        exit(-1);

    }

    // READ THE INPUT FILE
    // vector of samples used to train the SOM
    std::vector <double> Samples;
    // retrive the number of features for each read
    nElements = readSamplesfromFile(Samples, filePath);
    // device copy of samples
    double *d_Samples;
    cudaMalloc((void**)&d_Samples, sizeof(double) * Samples.size());
    cudaMemcpy(d_Samples, &Samples[0], sizeof(double) * Samples.size(), cudaMemcpyHostToDevice);


    // EXTRACTING THE MIN/MAX FROM SAMPLES(only used for random initialization)
    if (initializationType == 'r')
    {
	    // creating the thrust vector
	    thrust::device_ptr<double> dptr(d_Samples);
	    // extract the minimum
	    thrust::device_ptr<double> dresptr = thrust::min_element(dptr, dptr + Samples.size());
	    min_neuronValue = dresptr[0];
	    // extract maximum
	    dresptr = thrust::max_element(dptr, dptr + Samples.size());
	    max_neuronValue = dresptr[0];
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
    {
    	initialRadius = 1 + (max(nRows, nColumns)/2) * 2 / 3;
    }

    // total number of neurons in the SOM
    nNeurons = nRows * nColumns;
    // total length of the serialized matrix
    totalLength = nRows * nColumns * nElements;
    
    // number of block used in the computation
    nblocks = (nNeurons / 64) + 1;

    // CHECKING COMPUTABILITY ON CUDA
    if (nblocks >= 65535)
    {
    	std::cout << "Too many bocks generated, cannot run a kernel with so many blocks. Try to reduce the number of neurons" << std::endl;
    	exit(-1);
    }

    // CHECKING PARAMS COMPATIBILITY
    if(normalizedistance && ((distanceType=='t') || (distanceType=='m')))
    {
        std::cout << "NormalizeDistance option not avaiable with Manhattan or Tanimoto Distance" << std::endl;
        exit(-1);
    }

    // ALLOCATION OF THE STRUCTURES
    // host SOM representation(linearized matrix)
    double *h_Matrix = (double *)malloc(sizeof(double) * totalLength);
    // host distance array, store each neuron's distance, uset to search BMU
    double *h_Distance = (double *) malloc(sizeof(double) * nNeurons);
    // host distance history array,store BMU distance for each read of the iteration, used to compute accuracy
    double* h_DistanceHistory = (double *)malloc(sizeof(double) * nSamples);
    // device variables
    double *d_Matrix; 
    double *d_Distance;
    double* d_DistanceHistory;
    cudaMalloc((void**)&d_DistanceHistory, sizeof(double) * nSamples);
    cudaMalloc((void **)&d_Matrix, sizeof(double) * totalLength);
    cudaMalloc((void**)&d_Distance, sizeof(double) * nNeurons);


    // SOM INIZIALIZATION
    // generating random seed
    std::random_device rd;
    std::mt19937 e2(rd());
    // random values initializatoin between min and max values inluded in reads
    if (initializationType == 'r')
    {
        // uniform distribution of values
    	std::uniform_real_distribution<> dist(min_neuronValue, max_neuronValue);
	    for(int i = 0; i < totalLength; i++)
	    {
	    	h_Matrix[i] = dist(e2); 
	    }
    }
    // initialization with random reads choosen from the file passed
    else
    {   
        // uniform distribution of indexes
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

    // copy host SOM to device
    cudaMemcpy(d_Matrix, h_Matrix, sizeof(double) * totalLength, cudaMemcpyHostToDevice);

    // save the initial SOM to file
    if (debug | print)
        saveSOMtoFile("initialSOM.out", h_Matrix, nRows, nColumns, nElements);
    
	// inizializing actual values of lr, radius and accuracy
    lr = ilr;
    radius = initialRadius;
	accuracy = DBL_MAX;

    // debug print
    if(verbose | debug | print)
    {
        std::cout << "Running the program with " << nRows  << " rows, " << nColumns << " columns, " << nNeurons << " neurons, " << nElements << " features fot each read, " << ilr << " initial learning rate, " << flr << " final learning rate, " << accuracyTreshold<< " required accuracyTreshold, " << radius << " initial radius, ";
        std::cout << maxnIter << " max total iteration, " << distanceType << " distance type, " << normalizeFlag << " normalized, " << neighborsType << " neighbors function, ";
        std::cout << initializationType << " initialization teqnique, " << lattice << " lacttice, " << exponential << " type of decay, " << randomizeDataset << " randomized input, " << nSamples << " sample in the input file, " << nblocks << " blocks will be launched on the GPU" << std::endl;
    
    }

    // initializing indexes, used to shuffle the Samples vector at each new iteration
    int* randIndexes = new int[nSamples];
    for (int i = 0; i < nSamples; i++)
    {
    	randIndexes[i] = i;
    }


    // ITERATE UNTILL ACCURACY IS REACHED OR ITERATION ARE FINISHED
    while((accuracy >= accuracyTreshold) && (nIter < maxnIter))
    {
    	// randomize indexes of samples if required
    	if(randomizeDataset)
    		std::random_shuffle(&randIndexes[0], &randIndexes[nSamples-1]);

        // debug print
        if (debug | verbose)
        {
            std::cout << "Learning rate of this iteration is " << lr << std::endl;
            std::cout << "Radius of this iteration is " << radius << std::endl;
        }
            

        // ITERATE ON EACH SAMPLE TO FIND BMU
	    for(int s=0; s < nSamples ; s++)
        {
            // computing the Sample index for this iteration
            currentIndex = randIndexes[s]*nElements;
    
		    // parallel search of BMU launch
		    if (normalizeFlag)
            {
		    	switch(distanceType)
                {
		    		case 'e' : compute_distance_euclidean_normalized<<<nblocks, 64>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    		case 's' : compute_distance_sum_squares_normalized<<<nblocks, 64>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    		case 'm' : compute_distance_manhattan_normalized<<<nblocks, 64>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    		case 't' : compute_distance_tanimoto_normalized<<<nblocks, 64>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    	}
		    }
            else
            {
		    	switch(distanceType)
                {
		    		case 'e' : compute_distance_euclidean<<<nblocks, 64>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    		case 's' : compute_distance_sum_squares<<<nblocks, 64>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    		case 'm' : compute_distance_manhattan<<<nblocks, 64>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
				    case 't' : compute_distance_tanimoto<<<nblocks, 64>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    	}
		    }

		    cudaDeviceSynchronize();

            // chosing between CPU and GPU to find BMU 
            //TOFIX
            if(nNeurons < 400000)
            {
                // HOST IMPLEMENTATION TO FIND BMU
                // copy the distance array back to host
                cudaMemcpy(h_Distance, d_Distance, sizeof(double) * nNeurons, cudaMemcpyDeviceToHost);
                // minimum search
                BMU_distance = h_Distance[0];
                BMU_index = 0;
                for (int m = 1; m < nNeurons; m++){
                    if(BMU_distance > h_Distance[m]){
                        BMU_distance = h_Distance[m];
                        BMU_index = m;
                    }
                }
            }
            else
            {
                // DEVICE IMPLEMENTATION TO FIND BMU
                thrust::device_ptr<double> dptr2(d_Samples);
                // extract the minimum
                thrust::device_ptr<double> dresptr2 = thrust::min_element(dptr2, dptr2 + nNeurons);
                BMU_distance = dresptr2[0];
                BMU_index = dresptr2 - dptr2;
            }

            // compute BMU distance as requested and save in the history array
            if(!normalizedistance)
            {
                h_DistanceHistory[randIndexes[s]] = BMU_distance;
            }
            else
            {
                if(distanceType=='s')
                {
                    h_DistanceHistory[randIndexes[s]] = BMU_distance;
                }
                else if (distanceType=='e')
                {
                    BMU_distance = (BMU_distance) * (BMU_distance);
                    h_DistanceHistory[s] = BMU_distance;
                }
            }

			// debug print
		    if(debug)
			   std::cout << "The minimum distance is " << BMU_distance << " at position " << BMU_index << std::endl;


            // UPDATE THE NEIGHBORS
            // call the kernel function to update the device SOM
            if(radius == 0){
                update_BMU<<<1, 1>>>(d_Matrix, d_Samples, lr, currentIndex, nElements, BMU_index, neighborsType);
            }
            else{
                update_SOM<<<nblocks, 64>>>(d_Matrix, d_Samples, lr, currentIndex, nElements, BMU_index, nColumns, radius, nNeurons, neighborsType);
            }

            cudaDeviceSynchronize();      
        }


        // END OF EPOCH. UPDATING VALUES
        // computing accuracy as requested
        if(!normalizedistance){
        	cudaMemcpy(d_DistanceHistory, h_DistanceHistory, sizeof(double) * nSamples, cudaMemcpyHostToDevice);
            thrust::device_ptr<double> dptr(d_DistanceHistory);
            accuracy = thrust::reduce(dptr, dptr + nSamples);            
        }
        else
        {
            cudaMemcpy(d_DistanceHistory, h_DistanceHistory, sizeof(double) * nSamples, cudaMemcpyHostToDevice);
            thrust::device_ptr<double> dptr(d_DistanceHistory);
            accuracy = sqrt(thrust::reduce(dptr, dptr + nSamples) / nElements) / nSamples;  
        }

        // debug print
        if (verbose | debug)
        {
            std::cout << "Mean distance of this iteration is " << accuracy << std::endl;
        }
        
		// updating the counter iteration
		nIter ++;

        // updating radius and learning rate
        if (exponential== 'r' | exponential == 'b')
        	radius = (int) (initialRadius * exp(-(double)nIter/(sqrt(maxnIter))));
        else
            radius = (int) (initialRadius - (initialRadius) * ((double)nIter/maxnIter));

        if (exponential== 'l' | exponential == 'b')
        	lr = ilr * exp(- (double)nIter/sqrt(maxnIter)) + flr;
        else 
            lr = ilr - (ilr - flr) * ((double)nIter/maxnIter);
}

	std::cout << "\n\n TRAINING RESULTS" << std::endl;
	std::cout << "\nMean distance of the sample to the trained SOM is " << accuracy << std::endl;

    // save trainde SOM to file
    if (debug | print)
    {
        saveSOMtoFile("outputSOM.out", h_Matrix, nRows, nColumns, nElements);
    }

	//freeing all allocated memory
    cudaFree(d_Matrix);
    cudaFree(d_Samples);
    cudaFree(d_Distance);
    cudaFree(d_DistanceHistory);
    free(h_Matrix);
    free(h_Distance);
    free(randIndexes);
    free(h_DistanceHistory);
}