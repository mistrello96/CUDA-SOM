double gaussian(double distance, int radius){
    return exp(- (double)(distance * distance)/(double)(2 * radius * radius));
}

int bubble(double distance, int radius){
    if (distance <= radius)
        return 1;
    return 0;
}

double mexican_hat(double distance, int radius){
    return ((1 - (double)(distance*distance)/(double)(radius*radius)) * gaussian(distance, radius));
}

// function to check the avaiable memory
double checkFreeGpuMem()
{
	double free_m;
	size_t free_t,total_t;
	cudaMemGetInfo(&free_t,&total_t);
	free_m =(uint)free_t;

	return free_m;
}

int getnThreads(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
  
}

// returns the number of features per line
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
	
    // taken from Stack, TO FIX
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

    // kernel to update the SOM after the BMU has been found
__global__ void update_BMU(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex)
{
    int matrixindex = BMUIndex * nElements;
    for (int i = 0; i < nElements; i++){
        k_Matrix[matrixindex+i] = k_Matrix[matrixindex+i] + lr * (k_Samples[samplesIndex + i] - k_Matrix[matrixindex+i]); 
    }
}

__global__ void update_SOM(double* k_Matrix, double* k_Samples, double lr, int samplesIndex, int nElements, int BMUIndex, int nColumns, int radius, int nNeuron)
{
    int threadindex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadindex < nNeuron){
        int matrixindex = threadindex * nElements;
        int x = matrixindex / nColumns;
        int y = matrixindex % nColumns;
        int BMU_x = BMUIndex / nColumns;
        int BMU_y = BMUIndex % nColumns;
        int distance = 0;
        distance = (int)sqrtf((x - BMU_x) * (x - BMU_x) + (y - BMU_y) * (y - BMU_y));
        if (distance <= radius){
            double neigh = exp(- (double)(distance * distance)/(double)(2 * radius * radius));
            for (int i = 0; i < nElements; i++)
            {
                printf("%f\t%f", neigh, lr);
                k_Matrix[matrixindex+i] = k_Matrix[matrixindex+i] + neigh * lr * (k_Samples[samplesIndex + i] - k_Matrix[matrixindex+i]); 
            }
        }
    }
}


