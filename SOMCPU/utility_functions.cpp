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
void saveSOMtoFile(std::string filePath, double* matrix, int nRows, int nColumns, int nElements)
{
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
double gaussian(double distance, int radius)
{
    return exp(- (distance * distance)/(2 * radius * radius));
}

// bubble distance between two neurons
int bubble(double distance, int radius)
{
    if (distance <= radius)
        return 1;
    return 0;
}

// mexican hat distance between two neurons
double mexican_hat(double distance, int radius)
{
    return ((1 - (distance*distance)/(radius*radius)) * gaussian(distance, radius));
}

// compute distance between two neurons on a square toroidal map
int ComputeDistanceToroidal(int x1, int y1, int x2, int y2, int nRows, int nColumns)
{
    int a,b,c,d;
	(x1>x2) ? (a = x1, b = x2) : (a = x2, b = x1);
	(y1>y2) ? (c = y1, d = y2) : (c = y2, d = y1);
    int x = std::min(a-b, b + nRows - a);
    int y = std::min(c-d, d + nColumns - c);
    return sqrt(x*x + y*y);
}

// compute exagonal distance between two neurons	
int ComputeDistanceHexGrid(int x1, int y1, int x2, int y2)
{
    x1 = x1 - y1 / 2;
    x2 = x2 - y2 / 2;
    int dx = x2 - x1;
    int dy = y2 - y1;
    return std::max(std::max(abs(dx), abs(dy)), abs(dx+dy));
}

// compute distance between two neurons on a exagonal toroidal map
int ComputeDistanceHexGridToroidal(int x1, int y1, int x2, int y2, int nRows, int nColumns)
{
    if(x1 < x2)
    {
        if(y1 < y2)
        {
            int res = ComputeDistanceHexGrid(x1,y1,x2,y2);
            int tmp = ComputeDistanceHexGrid(x1,y1,x2-nRows,y2);
            if (res > tmp)
                res = tmp;
             tmp = ComputeDistanceHexGrid(x1,y1,x2,y2-nColumns);
             if (res > tmp)
                res = tmp;
             tmp = ComputeDistanceHexGrid(x1,y1,x2-nRows,y2-nColumns);
             if (res > tmp)
                res = tmp;
            return res;
        }
        else
        {
            int res = ComputeDistanceHexGrid(x1,y1,x2,y2);
            int tmp = ComputeDistanceHexGrid(x1,y1,x2-nRows,y2);
            if (res > tmp)
                res = tmp;
             tmp = ComputeDistanceHexGrid(x1,y1-nColumns,x2,y2);
             if (res > tmp)
                res = tmp;
             tmp = ComputeDistanceHexGrid(x1,y1-nColumns,x2-nRows,y2);
             if (res > tmp)
                res = tmp;
            return res;
            }
    }
    else
    {
        if(y1 < y2)
        {
            int res = ComputeDistanceHexGrid(x1,y1,x2,y2);
            int tmp = ComputeDistanceHexGrid(x1-nRows,y1,x2,y2);
            if (res > tmp)
                res = tmp;
             tmp = ComputeDistanceHexGrid(x1,y1,x2,y2-nColumns);
             if (res > tmp)
                res = tmp;
             tmp = ComputeDistanceHexGrid(x1-nRows,y1,x2,y2-nColumns);
             if (res > tmp)
                res = tmp;
            return res;
            }
        else
        {
            int res = ComputeDistanceHexGrid(x1,y1,x2,y2);
            int tmp = ComputeDistanceHexGrid(x1-nRows,y1,x2,y2);
            if (res > tmp)
                res = tmp;
             tmp = ComputeDistanceHexGrid(x1,y1-nColumns,x2,y2);
             if (res > tmp)
                res = tmp;
             tmp = ComputeDistanceHexGrid(x1-nRows,y1-nColumns,x2,y2);
             if (res > tmp)
                res = tmp;
            return res;
        }
    }
}