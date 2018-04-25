// returns the number of features per line. Save reads in samples vector
int readSamplesfromFile(std::vector<double>& samples, std::string filePath)
{
	int counter = 0;
	std::string line;
	std::ifstream file (filePath.c_str());
	if (file.is_open())
	{
		while (std::getline (file, line))
		{
			std::istringstream iss(line);
    		std::string element;
    		int tmp = 0;
    		while(std::getline(iss, element, '\t'))
    		{
    			tmp ++;
				samples.push_back(strtof((element).c_str(),0));
			}
			counter = tmp;
		}
		file.close();
		return counter;
	}
	else
	{
		std::cout << "Unable to open file";
		exit(-1);
	}
}

// same the SOM to a file. First two lines are nRows and Ncolumns. Neurons are \n separated, features are \t separated
void saveSOMtoFile(std::string filePath, double* matrix, int nRows, int nColumns, int nElements){
    std::ofstream myfile;
    myfile.open (filePath.c_str());
    myfile << "nRows \n" << nRows << "\n" << "nColumns \n" << nColumns << "\n";
    for (int i = 0; i < nRows*nColumns*nElements; i++)
    {
        myfile << matrix[i] << "\n";  
    }
    myfile.close();
}

// gaussian distance between two neurons
__device__
double gaussian(double distance, int radius)
{
    return exp(- (double)(distance * distance)/(double)(2 * radius * radius));
}

// bubble distance between two neurons
__device__
int bubble(double distance, int radius)
{
    if (distance <= radius)
        return 1;
    return 0;
}

// mexican hat distance between two neurons
__device__
double mexican_hat(double distance, int radius)
{
    return ((1 - (double)(distance*distance)/(double)(radius*radius)) * gaussian(distance, radius));
}
	
 __device__ 
int ComputeDistanceHexGrid(int x1, int y1, int x2, int y2)
{
    int du = x2 - x1;
    int dv = (y2 + x2 / 2) - (y1 + x1 / 2);
    if((du >= 0 && dv >= 0) || (du < 0 && dv < 0))
        return max(abs(du), abs(dv));
    else
        return abs(du) + abs(dv);
}

//TODO
 __device__ 
int ComputeDistanceHexGridToroidal(int x1, int y1, int x2, int y2)
{
    int du = x2 - x1;
    int dv = (y2 + x2 / 2) - (y1 + x1 / 2);
    if((du >= 0 && dv >= 0) || (du < 0 && dv < 0))
        return max(abs(du), abs(dv));
    else
        return abs(du) + abs(dv);
}

// run a benchmark to find out the minimum dimension of the input file to make GPU computation advantageous
void run_benchmark()
{
    // dimension of the array
    int dimension=5000;
    // flag used to increase the dimension if necessary
    bool again = true;
    while(again)
    {
        // create a host array and initialize to random number
        double *host_array = (double *) malloc(sizeof(double) * dimension);
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1, 1);
        for(int i = 0; i < dimension; i++)
        {
            host_array[i] = dist(e2); 
        }
        // create device array and copy the same values of the host array
        double *device_array;
        cudaMalloc((void**)&device_array, sizeof(double) * dimension);
        cudaMemcpy(device_array, host_array, sizeof(double) * dimension, cudaMemcpyHostToDevice);

        // start CPU computation
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        // minimum search
        double BMU_distance =0;
        for (int m = 0; m < dimension; m++)
        {
            BMU_distance += host_array[m];
        }
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        auto elapsedCPU = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count();
        std::cout << "CPU reduce on " << dimension << " array of double; " << elapsedCPU << " nanoseconds to compute " << BMU_distance <<std::endl;

        // start GPU computation
        std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();
        thrust::device_ptr<double> dptr(device_array);
        BMU_distance = thrust::reduce(dptr, dptr + dimension);

        std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
        auto elapsedGPU = (double)std::chrono::duration_cast<std::chrono::nanoseconds>( end2 - start2 ).count();
        std::cout << "GPU reduce on " << dimension << " array of double; " << elapsedGPU << " nanoseconds to compute " << BMU_distance <<std::endl;
        // check the results and increment if necessary
        if(elapsedCPU < elapsedGPU)
        {
            dimension = dimension * 2;
        }
        else
        {
            again = false;
        }

    }
    std::cout << "\n\nThe GPU computation is recommended on this system if (number_of_features * number_of_reads) is greater than " << dimension << "\n\n" << std::endl;
}
