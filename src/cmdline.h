/** @file cmdline.h
 *  @brief The header file for the command line option parser
 *  generated by GNU Gengetopt version 2.22.6
 *  http://www.gnu.org/software/gengetopt.
 *  DO NOT modify this file, since it can be overwritten
 *  @author GNU Gengetopt by Lorenzo Bettini */

#ifndef CMDLINE_H
#define CMDLINE_H

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h> /* for FILE */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef CMDLINE_PARSER_PACKAGE
/** @brief the program name (used for printing errors) */
#define CMDLINE_PARSER_PACKAGE "CudaSOM"
#endif

#ifndef CMDLINE_PARSER_PACKAGE_NAME
/** @brief the complete program name (used for help and version) */
#define CMDLINE_PARSER_PACKAGE_NAME "CudaSOM"
#endif

#ifndef CMDLINE_PARSER_VERSION
/** @brief the program version */
#define CMDLINE_PARSER_VERSION "1.1"
#endif

/** @brief Where the command line options are stored */
struct gengetopt_args_info
{
  const char *help_help; /**< @brief Print help and exit help description.  */
  const char *version_help; /**< @brief Print version and exit help description.  */
  char * learningmode_arg;	/**< @brief allows to choose between online training and batch training (default='o').  */
  char * learningmode_orig;	/**< @brief allows to choose between online training and batch training original value given at command line.  */
  const char *learningmode_help; /**< @brief allows to choose between online training and batch training help description.  */
  char * inputfile_arg;	/**< @brief PATH to the input file (default='./').  */
  char * inputfile_orig;	/**< @brief PATH to the input file original value given at command line.  */
  const char *inputfile_help; /**< @brief PATH to the input file help description.  */
  int nRows_arg;	/**< @brief allows to provide the number of rows of the neuron matrix (default='0').  */
  char * nRows_orig;	/**< @brief allows to provide the number of rows of the neuron matrix original value given at command line.  */
  const char *nRows_help; /**< @brief allows to provide the number of rows of the neuron matrix help description.  */
  int nColumns_arg;	/**< @brief allows to provide the number of columns of the neuron matrix (default='0').  */
  char * nColumns_orig;	/**< @brief allows to provide the number of columns of the neuron matrix original value given at command line.  */
  const char *nColumns_help; /**< @brief allows to provide the number of columns of the neuron matrix help description.  */
  double initial_learning_rate_arg;	/**< @brief allows to provide the initial learning rate for the training process (default='-1').  */
  char * initial_learning_rate_orig;	/**< @brief allows to provide the initial learning rate for the training process original value given at command line.  */
  const char *initial_learning_rate_help; /**< @brief allows to provide the initial learning rate for the training process help description.  */
  double final_learning_rate_arg;	/**< @brief allows to provide the final learning rate for the training process (default='0').  */
  char * final_learning_rate_orig;	/**< @brief allows to provide the final learning rate for the training process original value given at command line.  */
  const char *final_learning_rate_help; /**< @brief allows to provide the final learning rate for the training process help description.  */
  int iteration_arg;	/**< @brief number of times the dataset is presented to the SOM (default='-1').  */
  char * iteration_orig;	/**< @brief number of times the dataset is presented to the SOM original value given at command line.  */
  const char *iteration_help; /**< @brief number of times the dataset is presented to the SOM help description.  */
  int verbose_flag;	/**< @brief enables debug print (default=off).  */
  const char *verbose_help; /**< @brief enables debug print help description.  */
  int debug_flag;	/**< @brief enables advanced debug prints (default=off).  */
  const char *debug_help; /**< @brief enables advanced debug prints help description.  */
  int savedistances_flag;	/**< @brief saves distances between samples and the final SOM in a file called 'distances.out' (default=off).  */
  const char *savedistances_help; /**< @brief saves distances between samples and the final SOM in a file called 'distances.out' help description.  */
  int saveall_flag;	/**< @brief saves the input and output SOM in a file. It also saves distances between samples and the final SOM in a file called 'distances.out' (default=off).  */
  const char *saveall_help; /**< @brief saves the input and output SOM in a file. It also saves distances between samples and the final SOM in a file called 'distances.out' help description.  */
  char * savepath_arg;	/**< @brief PATH to saving folder (default='./').  */
  char * savepath_orig;	/**< @brief PATH to saving folder original value given at command line.  */
  const char *savepath_help; /**< @brief PATH to saving folder help description.  */
  int radius_arg;	/**< @brief allows to choose the initial radius used by the updating function (default='0').  */
  char * radius_orig;	/**< @brief allows to choose the initial radius used by the updating function original value given at command line.  */
  const char *radius_help; /**< @brief allows to choose the initial radius used by the updating function help description.  */
  char * distance_arg;	/**< @brief allows to choose different types of distance functions. Use e for euclidean, s for sum of sqares, m for manhattan or t for tanimoto (default='e').  */
  char * distance_orig;	/**< @brief allows to choose different types of distance functions. Use e for euclidean, s for sum of sqares, m for manhattan or t for tanimoto original value given at command line.  */
  const char *distance_help; /**< @brief allows to choose different types of distance functions. Use e for euclidean, s for sum of sqares, m for manhattan or t for tanimoto help description.  */
  char * neighbors_arg;	/**< @brief allows to specify the neighbour function used in the learning process. Use g for gaussian, b for bubble or m for mexican hat (default='g').  */
  char * neighbors_orig;	/**< @brief allows to specify the neighbour function used in the learning process. Use g for gaussian, b for bubble or m for mexican hat original value given at command line.  */
  const char *neighbors_help; /**< @brief allows to specify the neighbour function used in the learning process. Use g for gaussian, b for bubble or m for mexican hat help description.  */
  char * initialization_arg;	/**< @brief allows to specify how the initial weights of the SOM are initialized. Use r for random initialization or c for picking random vectors from the input file (default='c').  */
  char * initialization_orig;	/**< @brief allows to specify how the initial weights of the SOM are initialized. Use r for random initialization or c for picking random vectors from the input file original value given at command line.  */
  const char *initialization_help; /**< @brief allows to specify how the initial weights of the SOM are initialized. Use r for random initialization or c for picking random vectors from the input file help description.  */
  char * lattice_arg;	/**< @brief allows to choose what type of lattice is used for the SOM representation. Use s for square lattice or e for exagonal lattice (default='e').  */
  char * lattice_orig;	/**< @brief allows to choose what type of lattice is used for the SOM representation. Use s for square lattice or e for exagonal lattice original value given at command line.  */
  const char *lattice_help; /**< @brief allows to choose what type of lattice is used for the SOM representation. Use s for square lattice or e for exagonal lattice help description.  */
  int toroidal_flag;	/**< @brief allows to choose between planar topology and toroidal topology for edges of the SOM (default=off).  */
  const char *toroidal_help; /**< @brief allows to choose between planar topology and toroidal topology for edges of the SOM help description.  */
  int randomize_flag;	/**< @brief enables the randomization of the dataset. Before presentig the dataset to the SOM(each epoch), all entries are shuffled. (default=on).  */
  const char *randomize_help; /**< @brief enables the randomization of the dataset. Before presentig the dataset to the SOM(each epoch), all entries are shuffled. help description.  */
  char * exponential_arg;	/**< @brief enables the exponential decay of the learning rate and/or the radius. Use l for learning rate, r for radius or b for both (default='n').  */
  char * exponential_orig;	/**< @brief enables the exponential decay of the learning rate and/or the radius. Use l for learning rate, r for radius or b for both original value given at command line.  */
  const char *exponential_help; /**< @brief enables the exponential decay of the learning rate and/or the radius. Use l for learning rate, r for radius or b for both help description.  */
  int normalizedistance_flag;	/**< @brief enables the normalized mean distance. Not avaiable if Tanimoto distance is selected (default=off).  */
  const char *normalizedistance_help; /**< @brief enables the normalized mean distance. Not avaiable if Tanimoto distance is selected help description.  */
  int forceGPU_flag;	/**< @brief Runs all possible computation on GPU. Use only if the SOM number of neurons is is big enought(use the benchmark funtion to find out the minimum file size) (default=off).  */
  const char *forceGPU_help; /**< @brief Runs all possible computation on GPU. Use only if the SOM number of neurons is is big enought(use the benchmark funtion to find out the minimum file size) help description.  */
  int threadsperblock_arg;	/**< @brief allows to provide the number of threads per block (default='64').  */
  char * threadsperblock_orig;	/**< @brief allows to provide the number of threads per block original value given at command line.  */
  const char *threadsperblock_help; /**< @brief allows to provide the number of threads per block help description.  */
  int GPUIndex_arg;	/**< @brief allows to specify the device id of the GPU used for the computation (default='0').  */
  char * GPUIndex_orig;	/**< @brief allows to specify the device id of the GPU used for the computation original value given at command line.  */
  const char *GPUIndex_help; /**< @brief allows to specify the device id of the GPU used for the computation help description.  */
  int benchmark_flag;	/**< @brief Runs a benchmark to find out the minimum dimension of the input file to make GPU computation advantageous (default=off).  */
  const char *benchmark_help; /**< @brief Runs a benchmark to find out the minimum dimension of the input file to make GPU computation advantageous help description.  */
  
  unsigned int help_given ;	/**< @brief Whether help was given.  */
  unsigned int version_given ;	/**< @brief Whether version was given.  */
  unsigned int learningmode_given ;	/**< @brief Whether learningmode was given.  */
  unsigned int inputfile_given ;	/**< @brief Whether inputfile was given.  */
  unsigned int nRows_given ;	/**< @brief Whether nRows was given.  */
  unsigned int nColumns_given ;	/**< @brief Whether nColumns was given.  */
  unsigned int initial_learning_rate_given ;	/**< @brief Whether initial_learning_rate was given.  */
  unsigned int final_learning_rate_given ;	/**< @brief Whether final_learning_rate was given.  */
  unsigned int iteration_given ;	/**< @brief Whether iteration was given.  */
  unsigned int verbose_given ;	/**< @brief Whether verbose was given.  */
  unsigned int debug_given ;	/**< @brief Whether debug was given.  */
  unsigned int savedistances_given ;	/**< @brief Whether savedistances was given.  */
  unsigned int saveall_given ;	/**< @brief Whether saveall was given.  */
  unsigned int savepath_given ;	/**< @brief Whether savepath was given.  */
  unsigned int radius_given ;	/**< @brief Whether radius was given.  */
  unsigned int distance_given ;	/**< @brief Whether distance was given.  */
  unsigned int neighbors_given ;	/**< @brief Whether neighbors was given.  */
  unsigned int initialization_given ;	/**< @brief Whether initialization was given.  */
  unsigned int lattice_given ;	/**< @brief Whether lattice was given.  */
  unsigned int toroidal_given ;	/**< @brief Whether toroidal was given.  */
  unsigned int randomize_given ;	/**< @brief Whether randomize was given.  */
  unsigned int exponential_given ;	/**< @brief Whether exponential was given.  */
  unsigned int normalizedistance_given ;	/**< @brief Whether normalizedistance was given.  */
  unsigned int forceGPU_given ;	/**< @brief Whether forceGPU was given.  */
  unsigned int threadsperblock_given ;	/**< @brief Whether threadsperblock was given.  */
  unsigned int GPUIndex_given ;	/**< @brief Whether GPUIndex was given.  */
  unsigned int benchmark_given ;	/**< @brief Whether benchmark was given.  */

} ;

/** @brief The additional parameters to pass to parser functions */
struct cmdline_parser_params
{
  int override; /**< @brief whether to override possibly already present options (default 0) */
  int initialize; /**< @brief whether to initialize the option structure gengetopt_args_info (default 1) */
  int check_required; /**< @brief whether to check that all required options were provided (default 1) */
  int check_ambiguity; /**< @brief whether to check for options already specified in the option structure gengetopt_args_info (default 0) */
  int print_errors; /**< @brief whether getopt_long should print an error message for a bad option (default 1) */
} ;

/** @brief the purpose string of the program */
extern const char *gengetopt_args_info_purpose;
/** @brief the usage string of the program */
extern const char *gengetopt_args_info_usage;
/** @brief the description string of the program */
extern const char *gengetopt_args_info_description;
/** @brief all the lines making the help output */
extern const char *gengetopt_args_info_help[];

/**
 * The command line parser
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser (int argc, char **argv,
  struct gengetopt_args_info *args_info);

/**
 * The command line parser (version with additional parameters - deprecated)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param override whether to override possibly already present options
 * @param initialize whether to initialize the option structure my_args_info
 * @param check_required whether to check that all required options were provided
 * @return 0 if everything went fine, NON 0 if an error took place
 * @deprecated use cmdline_parser_ext() instead
 */
int cmdline_parser2 (int argc, char **argv,
  struct gengetopt_args_info *args_info,
  int override, int initialize, int check_required);

/**
 * The command line parser (version with additional parameters)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param params additional parameters for the parser
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_ext (int argc, char **argv,
  struct gengetopt_args_info *args_info,
  struct cmdline_parser_params *params);

/**
 * Save the contents of the option struct into an already open FILE stream.
 * @param outfile the stream where to dump options
 * @param args_info the option struct to dump
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_dump(FILE *outfile,
  struct gengetopt_args_info *args_info);

/**
 * Save the contents of the option struct into a (text) file.
 * This file can be read by the config file parser (if generated by gengetopt)
 * @param filename the file where to save
 * @param args_info the option struct to save
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_file_save(const char *filename,
  struct gengetopt_args_info *args_info);

/**
 * Print the help
 */
void cmdline_parser_print_help(void);
/**
 * Print the version
 */
void cmdline_parser_print_version(void);

/**
 * Initializes all the fields a cmdline_parser_params structure 
 * to their default values
 * @param params the structure to initialize
 */
void cmdline_parser_params_init(struct cmdline_parser_params *params);

/**
 * Allocates dynamically a cmdline_parser_params structure and initializes
 * all its fields to their default values
 * @return the created and initialized cmdline_parser_params structure
 */
struct cmdline_parser_params *cmdline_parser_params_create(void);

/**
 * Initializes the passed gengetopt_args_info structure's fields
 * (also set default values for options that have a default)
 * @param args_info the structure to initialize
 */
void cmdline_parser_init (struct gengetopt_args_info *args_info);
/**
 * Deallocates the string fields of the gengetopt_args_info structure
 * (but does not deallocate the structure itself)
 * @param args_info the structure to deallocate
 */
void cmdline_parser_free (struct gengetopt_args_info *args_info);

/**
 * Checks that all the required options were specified
 * @param args_info the structure to check
 * @param prog_name the name of the program that will be used to print
 *   possible errors
 * @return
 */
int cmdline_parser_required (struct gengetopt_args_info *args_info,
  const char *prog_name);

extern const char *cmdline_parser_distance_values[];  /**< @brief Possible values for distance. */
extern const char *cmdline_parser_neighbors_values[];  /**< @brief Possible values for neighbors. */
extern const char *cmdline_parser_initialization_values[];  /**< @brief Possible values for initialization. */
extern const char *cmdline_parser_lattice_values[];  /**< @brief Possible values for lattice. */
extern const char *cmdline_parser_exponential_values[];  /**< @brief Possible values for exponential. */


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* CMDLINE_H */
