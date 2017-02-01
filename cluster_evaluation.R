install.packages("clusterGeneration")
install.packages("MASS")
library(MASS)
library(clusterGeneration)
install.packages("plot3D")
library(plot3D)
install.packages("scatterplot3d")
library(scatterplot3d)
# Data Generation
# 1) Generate at least 3 well separated clusters in 4-Dimensions by using
either clusterGeneration package or built-in functions directly.
as.numeric(Sys.time())-> t
set.seed((t - floor(t)) * 1e8 -> seed) #change seed
data_v1 = clusterGeneration::genRandomClust(numClust=3, sepVal=0.01,
                                            numNonNoisy=4, clustszind=3, clustSizes=c(40,40,40))
data_v2 = clusterGeneration::genRandomClust(numClust=3, sepVal=0.1,
                                            numNonNoisy=4, clustszind=3, clustSizes=c(40,40,40))
data_v3 = clusterGeneration::genRandomClust(numClust=3, sepVal=0.5,
                                            numNonNoisy=4, clustszind=3, clustSizes=c(40,40,40))
data_v4 = clusterGeneration::genRandomClust(numClust=3, sepVal=0.7,
                                            numNonNoisy=4, clustszind=3, clustSizes=c(40,40,40))
data_v5 = clusterGeneration::genRandomClust(numClust=3, sepVal=0.9,
                                            numNonNoisy=4, clustszind=3, clustSizes=c(40,40,40))summary(data_v1$datList)
summary(data_v1$datList$test_1) # Summary of first cluster when sepVal=0.01
summary(data_v2$datList$test_1) # Summary of first cluster when sepVal=0.1
summary(data_v3$datList$test_1) # Summary of first cluster when sepVal=0.5
summary(data_v4$datList$test_1) # Summary of first cluster when sepVal=0.7
summary(data_v5$datList$test_1) # Summary of first cluster when sepVal=0.9
pairs(data_v1$datList$test_1) #scatter plot of each pair when sepVal=0.01 for
replicas test1
title(sub = "scatter plot of each pair when sepVal=0.01")
pairs(data_v2$datList$test_1) #scatter plot of each pair when sepVal=0.1 for
replicas test1
title(sub = "scatter plot of each pair when sepVal=0.1")
pairs(data_v3$datList$test_1) #scatter plot of each pair when sepVal=0.5 for
replicas test1
title(sub = "scatter plot of each pair when sepVal=0.5")
pairs(data_v4$datList$test_1) #scatter plot of each pair when sepVal=0.7 for
replicas test1
title(sub = "scatter plot of each pair when sepVal=0.7")
pairs(data_v5$datList$test_1) #scatter plot of each pair when sepVal=0.9 for
replicas test1
title(sub = "scatter plot of each pair when sepVal=0.9")
#2) Generate noise (random) data in 4-Dimensions
x=matrix(runif(100*4),100,4)
summary(x)
pairs(x)
title(sub = "noise data is generated in 4D ")
#3) Use or generate 3-Dimensional Swissroll data or generate 3-
Dimensional two circles which are not linearly separable
mydata = read.table("swissroll.dat")
mydata = swissroll
summary(mydata)
plot3D::scatter3D(mydata$V1,mydata$V2,mydata$V3) #scatter3D is with color
title(sub = "Swissroll data 3D scatter plot")
scatterplot3d::scatterplot3d(mydata$V1,mydata$V2,mydata$V3) #scatterplot3d
is black and white
title(sub = "Swissroll data 3D scatter plot (black and white)")#statistics
summary(mydata)
pairs(mydata)
#extra
iris
summary(iris)
plot(iris$Petal.Length, iris$Petal.Width, pch=21,
     bg=c("red","green3","blue")[unclass(iris$Species)], main="Edgar Anderson's Iris
Data"