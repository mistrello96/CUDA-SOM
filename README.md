# CUDA-SOM
A CUDA implementation of Self Organizing Maps for unsupervised learning.
The learning process is done by competitive learnig between neurons of the SOM.
The closest unit to the input vector is called Best Machine Unit (BMU) and his weigths vector is moved in the direction of the input vector.
Also the neighborhood of the BMU is moved in the same direction, but with lower magnitude, in function of the distance from the BMU.
This tool allows to specify lots of parameters used in the learning process, such as:
- Number of rows of the SOM
- Number of columns of the SOM
- Initial and final learning rate
- Number of iteration in the learning process
- Radius of the updating function
- Various type of distances for the BMU search (Euclidean, Sum of Squares, Manhattan, Tanimoto)
- Various neighbor function (gaussian, bubble, mexican hat)
- Two tipes of lattice for the neurons of the SOM (square or exagonal)
- Possibility to use a toroidal topology
- Possibility to decay the learing rate and/or the radius exponentially

The tool allows also to change some CUDA related parameters, such as:
- GPU index
- Number of threads per each block launched on the GPU
- An option to move all possible computation on the GPU

It is also included a small benchmark to identify the minimum size of your input file to make to make GPU computation advantageous

For all other functions included, please refer to the help menu included in the tool
--------------------------------------------------------

Here you can find some examples of the tool uses:
1) The tool will use the file provided as input, will train for 1000 iteration with a  learning rate that 
will linearly decay from 0.1 to 0.001. The distance between neurons and the input vector will be normalized. 
The size of the SOM and the radius will be estimated runtime with heuristcs. Neurons will be represented by exagones.

./CUDA-SOM.out -i /folder/folder/inputfile.txt -n 1000 -s 0.1 -f 0.001 --normalizedistance

2) The tool will use the file provided as input, will train for 5000 iteration with a learning rate that 
will linearly decay from 0.3 to 0.001. The size of the SOM is set to 200x200, the radius of the updating finction(bubble)
is set to 50, and will decrease exponentially. The distance between neurons and the input vector is computed 
using the sum of squares distance. Once the SOM is trained, the distances between the input file and the SOM will be
saved to a file. The GPU used for the computation will be the one with index 2, 96 threads per block will be launched.

./CUDA-SOM.out -i /folder/folder/inputfile.txt -n 5000 -s 0.3 -f 0.001 -x 200 -y 200 --savedistances -r 50 --distance=s --neighbors=b --toroidal --exponential=r --GPUIndex=2 --threadsperblock=96

