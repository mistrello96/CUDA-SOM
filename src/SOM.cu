#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <float.h>
#include <fstream>

#include "utility_functions.h"
#include "distance_kernels.h"
#include "update_kernels.h"
#include "cmdline.h"

int main(int argc, char **argv)
{
	// reading passed parameters
	gengetopt_args_info ai;
    if (cmdline_parser (argc, argv, &ai) != 0)
    {
        exit(1);
    }
    // checking if benchmark mode is set. If so, run the benchmark function and terminate the program
    if(ai.benchmark_flag)
    {
        run_benchmark();
        exit(1);
    }

	// INIZIALIZING VARIABLES WITH DEFAULT VALUES
	// path of the input file
	std::string filePath = ai.inputfile_arg;
	// verbose
    bool verbose = ai.verbose_flag;
    // advanced debug
    bool debug = ai.debug_flag;
    // save SOM to file
    bool saveall = ai.saveall_flag;
    // save distances to file
    bool savedistances = ai.savedistances_flag;
    // PATH to the folder where saved files will be placed
    std::string savePath = ai.savepath_arg;
    // number of rows in the matrix
    int nRows = ai.nRows_arg;
    // number of columns in the matrix
    int nColumns = ai.nColumns_arg;
    // initial learning rate
    double ilr = ai.initial_learning_rate_arg;
    // final learning rate
    double flr = ai.final_learning_rate_arg;
    // number of iterations(epochs)
    int maxnIter = ai.iteration_arg;
    // initial radius for the update function
    double initialRadius = ai.radius_arg;
    // type of distance
    char distanceType = ai.distance_arg[0];
    // type of neighbour function
    char neighborsType = ai.neighbors_arg[0];
    // type of initialization
    char initializationType = ai.initialization_arg[0];
    // number of threads per block 
    int tpb = ai.threadsperblock_arg;
    // type of lattice
    char lattice = ai.lattice_arg[0];
    // exponential decay for radius and/or learning rate
    char exponential = ai.exponential_arg[0];
    // dataset presentation method
    bool randomizeDataset = ai.randomize_flag;
    // normalization of the average distance of each iteration (epoch)
    bool normalizedistance = ai.normalizedistance_flag;
    // toroidal topology
    bool toroidal = ai.toroidal_flag;
    // move all computation on GPU
    bool forceGPU = ai.forceGPU_flag;
    // device id used for computation
    int deviceIndex = ai.GPUIndex_arg;
    // counter of epochs/iterations
    int nIter = 0;
    
    // DECLARATION OF USEFULL VARIABLES
    // min and max values of neurons (used for random initialization)
    double min_neuronValue, max_neuronValue;
    // number of lines/samples in the input file
    int nSamples;
    // total number of neurons in the SOM
    int nNeurons;
    // total length of the matrix vector (nNeurons*nElements)
    int totalLength;
    // number of features in each sample
    int nElements;
    // number of blocks that need to be launched on the GPU
    int nblocks;
    // learning rate of the ongoing iteration
    double lr;
    // radius of the ongoing iteration
    double radius;
    // accuracy of the ongoing iteration
	double accuracy;
	// index of the sample picked for the iteration
    int currentIndex;
    // BMU distance to the sample
    double BMU_distance;
    // BMU index
    unsigned int BMU_index;
    // file used to save distances of samples to their BMU on the last epoch
    std::ofstream distancesfile;

    //checking the required params
    if(ilr == -1 || maxnIter == -1)
    {
        std::cout << "./a.out: '--initial_learning_rate' ('-s') option required" << std::endl;
        std::cout << "./a.out: '--iteration' ('-n') option required " << std::endl;
        exit(-1);
    }

    // checking the availability of the device required. If available, set the device for the computation
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    if(deviceIndex < devicesCount)
    {
        cudaSetDevice(deviceIndex);
    }
    else
    {
        std::cout << "Device not avaiable" << std::endl;
        exit(-1);        
    }

    // READ THE INPUT FILE
    // vector of samples used to train the SOM
    std::vector <double> samples;
    // retrive the number of features in each sample
    nElements = readSamplesfromFile(samples, filePath);
    // copy samples on the device
    double *d_Samples;
    cudaMalloc((void**)&d_Samples, sizeof(double) * samples.size());
    cudaMemcpy(d_Samples, &samples[0], sizeof(double) * samples.size(), cudaMemcpyHostToDevice);


    // EXTRACTING THE MIN/MAX FROM SAMPLES(only used for random initialization)
    if (initializationType == 'r')
    {
	    // creating the thrust vector
	    thrust::device_ptr<double> dptr(d_Samples);
	    // extract the minimum
	    thrust::device_ptr<double> dresptr = thrust::min_element(dptr, dptr + samples.size());
	    min_neuronValue = dresptr[0];
	    // extract maximum
	    dresptr = thrust::max_element(dptr, dptr + samples.size());
	    max_neuronValue = dresptr[0];
	}

    // COMPUTE VALUES FOR THE SOM INITIALIZATION
    // retrive the number of samples
    nSamples = samples.size() / nElements;

    // estimate the neuron number if not given(using heuristic)
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
    
    // number of blocks used in the computation
    nblocks = (nNeurons / tpb) + 1;

    // CHECKING COMPUTABILITY ON CUDA
    if (nblocks >= 65535)
    {
    	std::cout << "Too many bocks generated, cannot run a kernel with so many blocks. Try to reduce the number of neurons" << std::endl;
    	exit(-1);
    }

    // CHECKING PARAMS COMPATIBILITY
    if(normalizedistance && (distanceType=='t'))
    {
        std::cout << "NormalizeDistance option not avaiable with Tanimoto Distance" << std::endl;
        exit(-1);
    }

    // ALLOCATION OF THE STRUCTURES
    // host SOM representation(linearized matrix)
    double *h_Matrix = (double *)malloc(sizeof(double) * totalLength);
    // host array of the distances, it stores each neuron distance from the sample, used to search BMU
    double *h_Distance = (double *) malloc(sizeof(double) * nNeurons);
    // host distance history array, it stores BMU distance for each sample of the iteration, used to compute accuracy
    double h_DistanceHistory = 0;
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
    // random values initialization between min and max values included in samples
    if (initializationType == 'r')
    {
        // uniform distribution of values
    	std::uniform_real_distribution<> dist(min_neuronValue, max_neuronValue);
	    for(int i = 0; i < totalLength; i++)
	    {
	    	h_Matrix[i] = dist(e2); 
	    }
    }
    // initialization with random samples choosen from the input file
    else
    {   
        // uniform distribution of indexes
    	std::uniform_int_distribution<> dist(0, nSamples);
	    for (int i = 0; i < nNeurons; i++)
	    {
	        int r = dist(e2);
	        for (int k = i * nElements, j = 0; j < nElements; k++, j++)
	        {
	             h_Matrix[k] = samples[r*nElements + j];
	        }
	    }
	}

    // copy host SOM to device
    cudaMemcpy(d_Matrix, h_Matrix, sizeof(double) * totalLength, cudaMemcpyHostToDevice);

    // save the initial SOM to file
    if (debug | saveall)
    {
        saveSOMtoFile(savePath + "/initialSOM.out", h_Matrix, nRows, nColumns, nElements);
    }
    
	// inizializing values of lr, radius and accuracy for the first iteration
    lr = ilr;
    radius = initialRadius;
	accuracy = DBL_MAX;

    // debug print
    if(verbose | debug)
    {
        std::cout << "Running the program with " << nRows  << " rows, " << nColumns << " columns, " << nNeurons << " neurons, " 
        << nElements << " features fot each sample, " << ilr << " initial learning rate, " << flr << " final learning rate, "
        << radius << " initial radius, " << maxnIter << " max total iteration, " << distanceType << " distance type, " 
        << neighborsType << " neighbors function, " << initializationType << " initialization teqnique, " 
        << lattice << " lacttice, " << exponential << " type of decay, " << randomizeDataset << " randomized input, " 
        << nSamples << " sample in the input file, " << nblocks << " blocks will be launched on the GPU" << std::endl;
    }

    // initializing index array, used to shuffle the sample vector at each new iteration
    int* randIndexes = new int[nSamples];
    for (int i = 0; i < nSamples; i++)
    {
    	randIndexes[i] = i;
    }


    // ITERATE UNTILL MAXNITER IS REACHED
    while(nIter < maxnIter)
    {
    	// randomize sample indexes if required
    	if(randomizeDataset)
        {
    		std::random_shuffle(&randIndexes[0], &randIndexes[nSamples-1]);
        }

        // debug print
        if (debug | verbose)
        {
            std::cout << "Learning rate of this iteration is " << lr << std::endl;
            std::cout << "Radius of this iteration is " << radius << std::endl;
        }

        // open the file used for saving distances of samples during the last iteration
        if(nIter == (maxnIter-1) && (savedistances || saveall))
        {   
            distancesfile.open(savePath + "/distances.out");
        }
            

        // ITERATE ON EACH SAMPLE TO FIND CORRESPONDING BMU
	    for(int s=0; s < nSamples ; s++)
        {
            // computing the sample index for ongoing iteration
            currentIndex = randIndexes[s]*nElements;
    		
    		// device computation of distance between neurons and sample
		    switch(distanceType)
            {
		    	case 'e' : compute_distance_euclidean<<<nblocks, tpb>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    	case 's' : compute_distance_sum_squares<<<nblocks, tpb>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    	case 'm' : compute_distance_manhattan<<<nblocks, tpb>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
				case 't' : compute_distance_tanimoto<<<nblocks, tpb>>>(d_Matrix, d_Samples, currentIndex, d_Distance, nNeurons, nElements); break;
		    }

		    cudaDeviceSynchronize();

            // chosing between CPU and GPU to find BMU(CPU used by default, since the nNeurons array is not so big)
            if(forceGPU)
            {
                // DEVICE IMPLEMENTATION TO FIND BMU
                thrust::device_ptr<double> dptr2(d_Distance);
                // extract the minimum
                thrust::device_ptr<double> dresptr2 = thrust::min_element(dptr2, dptr2 + nNeurons);
                BMU_distance = dresptr2[0];
                BMU_index = dresptr2 - dptr2;
            }
            else
            {
                // HOST IMPLEMENTATION TO FIND BMU
                // copy the distance array back to host
                cudaMemcpy(h_Distance, d_Distance, sizeof(double) * nNeurons, cudaMemcpyDeviceToHost);
                // minimum search
                BMU_distance = h_Distance[0];
                BMU_index = 0;
                for (int m = 1; m < nNeurons; m++)
                {
                    if(BMU_distance > h_Distance[m])
                    {
                        BMU_distance = h_Distance[m];
                        BMU_index = m;
                    }
                }
            }

            // debug
		    if(debug)
            {
			   std::cout << "The minimum distance is " << BMU_distance << " at position " << BMU_index << std::endl;
            }

            // if  required, during the last iteration , save distance to file
            if (nIter == (maxnIter-1) && (savedistances || saveall))
            {
                distancesfile << "The minimum distance of the "<< currentIndex << " sample is " << BMU_distance << " at position " << BMU_index << "\n";
            }

            // compute BMU distance history as required and save it in the history array
            if(!normalizedistance)
            {
                h_DistanceHistory += BMU_distance;
            }
            else
            {
                if(distanceType=='s')
                {
                    h_DistanceHistory += BMU_distance;
                }
                else if (distanceType=='e' | distanceType=='m')
                {
                    BMU_distance = (BMU_distance) * (BMU_distance);
                    h_DistanceHistory += BMU_distance;
                }
            }


            // UPDATE THE NEIGHBORS
            // call the kernel function to update the device SOM
            if(radius == 0)
            {
                update_BMU<<<1, 1>>>(d_Matrix, d_Samples, lr, currentIndex, nElements, BMU_index);
            }
            else
            {
                if (toroidal)
                {
                    if(lattice == 's')
                        update_SOM_toroidal<<<nblocks, tpb>>>(d_Matrix, d_Samples, lr, currentIndex, nElements, BMU_index, nRows, nColumns, radius, nNeurons, neighborsType);
                    else
                        update_SOM_exagonal_toroidal<<<nblocks, tpb>>>(d_Matrix, d_Samples, lr, currentIndex, nElements, BMU_index, nRows, nColumns, radius, nNeurons, neighborsType);
                }
                else
                {
                    if(lattice == 's')
                        update_SOM<<<nblocks, tpb>>>(d_Matrix, d_Samples, lr, currentIndex, nElements, BMU_index, nColumns, radius, nNeurons, neighborsType);
                    else
                        update_SOM_exagonal<<<nblocks, tpb>>>(d_Matrix, d_Samples, lr, currentIndex, nElements, BMU_index, nColumns, radius, nNeurons, neighborsType);
                }
            }

            cudaDeviceSynchronize();      
        }


        // END OF EPOCH. UPDATING VALUES
        // computing accuracy as required
        if(!normalizedistance)
        {
            accuracy = h_DistanceHistory / nSamples;
            h_DistanceHistory = 0;            
        }
        else
        {
            accuracy = sqrt(h_DistanceHistory / nElements) / nSamples;
            h_DistanceHistory = 0;  
        }

        // debug print
        if (verbose | debug)
        {
            std::cout << "Mean distance of this iteration is " << accuracy << std::endl;
        }
        
		// updating the counter of iterations
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
    if (debug | saveall)
    {
    	cudaMemcpy(h_Matrix, d_Matrix, sizeof(double) * totalLength, cudaMemcpyDeviceToHost);
        saveSOMtoFile(savePath + "/outputSOM.out", h_Matrix, nRows, nColumns, nElements);
    }

	//freeing all allocated memory
    cudaFree(d_Matrix);
    cudaFree(d_Samples);
    cudaFree(d_Distance);
    cudaFree(d_DistanceHistory);
    free(h_Matrix);
    free(h_Distance);
    free(randIndexes);

}