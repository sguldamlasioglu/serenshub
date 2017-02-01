#PCA
#For each of the data generated (well separated clusters, random data, not linearly separable data)
#do the following:
#1) Determine the eigenvalues and eigenvectors.
#2) Plot the scree graph and proportion of variance.
#3) Plot directions of the all of the principal components.
#4) Plot the transformed data onto to the new coordinate system by using the first two principal
components.
## 1) Generate at least 3 well separated clusters in 4-Dimensions by using either
## clusterGeneration package or built-in functions directly.
as.numeric(Sys.time())-> t
set.seed((t - floor(t)) * 1e8 -> seed) #change seed
# generated well clustered data in 4D
data_v5 = clusterGeneration::genRandomClust(numClust=3, sepVal=0.9, numNonNoisy=4,
                                            clustszind=3, clustSizes=c(40,40,40))
# selected well clustered data in 4D
data <- data_v5$datList$test_1
# a) principal component analysis
pca <- princomp(data, cor = TRUE, scores = TRUE)
names(pca)summary(pca)
pca$scores
# eigenvalues and eigenvectors
eigen_val <- pca$sdev^2
eigen_val
eigen_vec <- pca$loadings
eigen_vec
# b) plot the scree plot and proportion of variance
screeplot(pca, type = 'l')
screeplot(pca)
# c) plotting directions of the all of the principal components
biplot (pca , scale =0)
# d) plotting the transformed data onto to the new coordinate system by using the first two principal
components.
#center and scale the row data
df.scaled <- scale(data, center = TRUE, scale = TRUE)
#select eigenvectors according to first two principal components
new_eigen_vec <- eigen_vec[,1:2]
new_eigen_vec
#Transpose eigeinvectors
eigenvectors.t <- t(new_eigen_vec)
eigenvectors.t
# Transpose the adjusted data
df.scaled.t <- t(df.scaled)
df.scaled.t
# The new dataset
df.new <- eigenvectors.t %*% df.scaled.t
# Transpose new data ad rename columns
df.new <- t(df.new)
colnames(df.new) <- c("PC1", "PC2")
head(df.new)
# plot the transformed data on to the new coord system
plot (df.new)
title(sub = "transformed data on to the new coord system")##################################################################################
#########################
#2) Generate noise (random) data in 4-Dimensions
random_data =matrix(runif(100*4),100,4)
pca <- princomp(random_data, cor = TRUE, scores = TRUE)
summary(pca)
# eigenvalues and eigenvectors
# eigenvalues and eigenvectors
eigen_val <- pca$sdev^2
eigen_val
eigen_vec <- loadings(pca)
eigen_vec
# b) plot the scree plot and proportion of variance
screeplot(pca, type = 'l')
title(sub = "screeplot")
screeplot(pca)
title(sub = "screeplot")
# c) plotting directions of the all of the principal components
biplot (pca , scale =0)
title(sub = "biplot")
# d) plotting the transformed data onto to the new coordinate system by using the first two principal
components.
#center and scale the row data
df.scaled <- scale(random_data, center = TRUE, scale = TRUE)
#select eigenvectors according to first two principal components
new_eigen_vec <- eigen_vec[,1:2]
new_eigen_vec
#Transpose eigeinvectors
eigenvectors.t <- t(new_eigen_vec)
eigenvectors.t
# Transpose the adjusted data
df.scaled.t <- t(df.scaled)
df.scaled.t
# The new dataset
df.new <- eigenvectors.t %*% df.scaled.tdf.new
# Transpose new data ad rename columns
df.new <- t(df.new)
colnames(df.new) <- c("PC1", "PC2")
head(df.new)
# plot the transformed data on to the new coord system
plot (df.new)
title(sub = "transformed data on to the new coord system")
##################################################################################
#########################
#3) Use or generate 3-Dimensional Swissroll data or generate 3-Dimensional two circles which are
not linearly separable
non_linear = swissroll
pca3 <- princomp(non_linear, cor = FALSE, scores = TRUE)
summary(pca3)
# eigenvalues and eigenvectors
# eigenvalues and eigenvectors
eigen_val3 <- pca3$sdev^2
eigen_val3
eigen_vec3 <- loadings(pca3)
eigen_vec3
# b) plot the scree plot and proportion of variance
screeplot(pca3, type = 'l')
title(sub = "screeplot")
screeplot(pca3)
title(sub = "screeplot")
# c) plotting directions of the all of the principal components
biplot (pca3 , scale =0)
title(sub = "biplot")
# d) plotting the transformed data onto to the new coordinate system by using the first two principal
components.#center and scale the row data
df.scaled <- scale(non_linear, center = TRUE, scale = TRUE)
#select eigenvectors according to first two principal components
new_eigen_vec3 <- eigen_vec3[,1:2]
new_eigen_vec3
#Transpose eigeinvectors
eigenvectors.t <- t(new_eigen_vec3)
eigenvectors.t
# Transpose the adjusted data
df.scaled.t <- t(df.scaled)
df.scaled.t
# The new dataset
df.new <- eigenvectors.t %*% df.scaled.t
# Transpose new data ad rename columns
df.new <- t(df.new)
colnames(df.new) <- c("PC1", "PC2")
head(df.new)
# plot the transformed data on to the new coord system
plot (df.new)
title(sub = "transformed data on to the new coord system")