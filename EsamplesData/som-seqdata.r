# mandatory includes 
require("kohonen")
require("cluster")
require("factoextra")
require("NbClust")

# setwd("D:\\Ricerca\\clustering")

# heuristics from som toolbox
optimal_som <- function(data) {
  return (sqrt(5*nrow(data)^0.54321)); 
}

get_best_k <- function(seqdata) {
  a <- get_best_k_gapstat(seqdata);
  b <- get_best_k_silhouette(seqdata);
  print(a);
  print(b);
#  return( (a+b)/2 );
  return (b); # more sbracativ
}


get_best_k_gapstat <- function(seqdata) {
  ret <- fviz_nbclust(seqdata, hcut, method = "gap_stat",hc_method = "complete");
  for (i in 1:9) {
    if (ret$data$gap[i]>ret$data$gap[i+1])  {
      return(i);
    }
  }
  return(10);
}

get_best_k_silhouette <- function(seqdata) {
  ret <- fviz_nbclust(seqdata, hcut, method = "silhouette",hc_method = "ward.D2");
  return (which(ret$data$y==max(ret$data$y)));
}

get_best_k_wss <- function(seqdata) {
  ret <- fviz_nbclust(seqdata, hcut, method = "wss", hc_method = "ward.D2");
  return (which(ret$data$y==max(ret$data$y)));
}



#setwd("C:\\Users\\aresio\\Documents\\PythonCode\\clustering")

# read and format files
input_file = "bigData.txt" # dati raw
input_file = "leucemia_scaled.txt" # dati raw
input_file = "melanoma_scaled.txt" # dati raw
test_name = "Leukemia"
test_name = "Melanoma"
seqdata_original <- read.csv(input_file, sep="\t", dec=".", header=FALSE)
seqdata <- as.matrix(seqdata_original)
#best_clustering <- get_best_k(seqdata);
#best_clustering <- NbClust(seqdata, distance="euclidean", min.nc=2, max.nc=10, method="complete", index="all")

#for (best_clustering in c(2,3,4,5,6,7,8,9,10)) {
  
  # main plot
  D <- optimal_som(seqdata);
  #debug(supersom);
  system.time(om_model <- supersom(seqdata, rlen=5, alpha=c(0.1, 0.001), grid=somgrid(D,D,"hexagonal",toroidal = F),keep.data=T, dist.fcts="euclidean", cores = 1))
  coolBlueHotRed <- function(n, alpha = 1) {rainbow(n, end=4/6, alpha=alpha)[n:1]}
  plot(som_model, type="dist.neighbours",  shape="straight", main=paste("Neighbours distances - ", test_name), palette.name = coolBlueHotRed)
 # cluster = hclust(object.distances(som_model, "codes"))
  # som.hc <- cutree(cluster, k=2);
  #som.hc <- cutree(cluster, k=2); 
  #add.cluster.boundaries(som_model, som.hc);
  
  # extract cluster data
  #write(NULL, file = paste("outputnew_", test_name, "k", best_clustering));
  #for (c in 1:best_clustering)  {
   # name <- paste("cluster", c);
    #print (name);
   # write(name, file = paste("outputnew_", test_name, "k", best_clustering), append=T);
  #  tmp <- which(som.hc==c);
    #print (tmp);
    #result_extr <- which(som_model$unit.classif %in% tmp);
    #print (result_extr);
   # write(result_extr, file = paste("outputnew_", test_name, "k", best_clustering), append=T);
  #}


#}

######################################################################################

# extract codes for each neuron
#for (n in 1:nrow(seqdata)) {
#  cat("Element number: ");  print(n);
#  cat("Neuron (unit classificator): "); print(som_model$unit.classif[n]);
#  for (m in 1:ncol(seqdata)) {
#    cat(seqdata_original[n,m]) ;
#    cat(", ");
#  }
#  cat("\n\n");
#}

# further tests
#plot(som_model, type="mapping", shape="straight", main=paste("Mapping - ", test_name))
plot(som_model, type="changes")
#plot(som_model, type="count",  shape="straight", main=paste("Counts - ", test_name))
