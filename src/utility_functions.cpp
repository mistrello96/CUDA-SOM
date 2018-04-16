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
double gaussian(double distance, int radius){
    return exp(- (double)(distance * distance)/(double)(2 * radius * radius));
}

// bubble distance between two neurons
__device__
int bubble(double distance, int radius){
    if (distance <= radius)
        return 1;
    return 0;
}

// mexican hat distance between two neurons
__device__
double mexican_hat(double distance, int radius){
    return ((1 - (double)(distance*distance)/(double)(radius*radius)) * gaussian(distance, radius));
}
	
 __device__ 
int ComputeDistanceHexGrid(int ax, int ay, int bx, int by)
{
    // compute distance as we would on a normal grid
    int xdist = ax - bx;
    int ydist = ay - by;

    // compensate for grid deformation
    // grid is stretched along (-n, n) line so points along that line have
    // a distance of 2 between them instead of 1

    // to calculate the shortest path, we decompose it into one diagonal movement(shortcut)
    // and one straight movement along an axis

    int lesserCoord = abs(xdist) < abs(ydist) ? abs(xdist) : abs(ydist);
    int diagx = (xdist < 0) ? -lesserCoord : lesserCoord; // keep the sign 
    int diagy = (ydist < 0) ? -lesserCoord : lesserCoord; // keep the sign

    // one of x or y should always be 0 because we are calculating a straight
    // line along one of the axis
    int strx = xdist - diagx;
    int stry = ydist - diagy;

    // calculate distance
    int straightDistance = abs(strx) + abs(stry);
    int diagonalDistance = abs(diagx);

    // if we are traveling diagonally along the stretch deformation we double
    // the diagonal distance
    if ( (diagx > 0 && diagy < 0) || (diagx < 0 && diagy > 0) )
    {
        diagonalDistance *= 2;
    }

    return straightDistance + diagonalDistance;
}

void run_benchmark(){
    int dimension=5000;
    bool again = true;
    while(again)
    {
        double *host_array = (double *) malloc(sizeof(double) * dimension);
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1, 1);
        for(int i = 0; i < dimension; i++)
        {
            host_array[i] = dist(e2); 
        }
        
        double *device_array;
        cudaMalloc((void**)&device_array, sizeof(double) * dimension);
        cudaMemcpy(device_array, host_array, sizeof(double) * dimension, cudaMemcpyHostToDevice);


        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        // minimum search
        double BMU_distance =0;
        for (int m = 0; m < dimension; m++){
            BMU_distance += host_array[m];
        }
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        auto elapsedCPU = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count();
        std::cout << "CPU reduce on " << dimension << " array of double; " << elapsedCPU << " nanoseconds to compute " << BMU_distance <<std::endl;


        std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();
        thrust::device_ptr<double> dptr(device_array);
        BMU_distance = thrust::reduce(dptr, dptr + dimension);

        std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
        auto elapsedGPU = (double)std::chrono::duration_cast<std::chrono::nanoseconds>( end2 - start2 ).count();
        std::cout << "GPU reduce on " << dimension << " array of double; " << elapsedGPU << " nanoseconds to compute " << BMU_distance <<std::endl;
        if(elapsedCPU < elapsedGPU)
            dimension = dimension * 2;
        else
            again = false;

    }
    std::cout << "\n\nThe GPU computation is recommended on this system if (number_of_features * number_of_reads) is greater than " << dimension << "\n\n" << std::endl;
}
