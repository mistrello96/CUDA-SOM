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

/*
void loadMatrixfromFile(std::string filePath, double* h_Matrix, int* &nRows, int* &nColumns){
    int counter = 0;
    std::string line;
    std::ifstream file (filePath.c_str());
    if (file.is_open())
    {
        std::getline (file, line);
        std::getline (file, line);
        *nRows = strtoi((line).c_str(),0);
        std::getline (file, line);
        std::getline (file, line);
        *nRows = strtoi((line).c_str(),0);
        while (std::getline (file, line))
        {
            h_Matrix[counter] = strtoi((line).c_str(),0);
            counter ++;
        }
        file.close();
        return counter;
    }
}
*/

