# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:17:29 2023

@author: jverl
"""

import numpy as np

X = np.genfromtxt("clusteringData.csv", delimiter=',') # loading data

######################################################
# Initialising parameters and arrays to hold values
######################################################

b = 1000 # setting batch size

k=6 # setting number of clusters

data_clusters = np.zeros(shape=(1,len(X))) # intialising array to hold all clusters

t = 20 # setting num iterations

num_restarts = 20

distortions = np.zeros(shape=(1,num_restarts))

#################################
# functions
#################################

def create_centroids(data, k): # Picks k random rows from data 
    indexes = np.random.choice(len(data), size=k, replace = False) # picks k distinct random indexes
    centroids = data[indexes] # assigns centroids based on indexes
    return centroids

# creates a batch of size b
def create_batch(X, M_indexes): # takes a set of b random indexes and dataset 
    M = X[M_indexes] # selects mini-batch of random rows from the data of size b
    return M

# Finds distance between a single query point and a data set with same number of columns
def calculate_distance(X,point):
    diffs=np.subtract(X,point)  # finds the difference by element between each 
                                # row of data and a single query point 
    squared_diffs=np.power(diffs,2) # Finds squared differences by row
    sum_squares=np.sum(squared_diffs, axis=1) # Sums up to find squared euclidian distance
    return sum_squares

# Assigns centroids based on the minimum distance 
# between each point in a dataset or mini-batch 
def assign_centroids(X, C, k):
    distance_array = np.zeros(shape=(k,len(X))) # creating array to hold distances
    for i in range(k): # calculates distances through k centroids
        distance_array[i] = calculate_distance(X, C[i])
        
    cluster_indices = np.argmin(distance_array,axis=0) 
    # selects the minimum distance for each point
    # to create an array of cluster indexes
    return cluster_indices


# Calculates gradient step for each iteration
def gradient_step(m_batch_clusters, M, C, N):
    for i in range(b): # loops through each point in the mini-batch
        cluster_index = m_batch_clusters[i] # saves sample to variable cluster_index
        centre = C[cluster_index] # finds centroid nearest the sample
        N[m_batch_clusters[i],] += 1 # updates the number of points belonging to the cluster
        N_sample = N[m_batch_clusters[i],]
        lr = 1/N_sample # calculates inverse for learning rate
        C[cluster_index] = (1 - lr)*centre + lr*M[i] # updates cluster according to gradient step
    return C

# finds sum of squared distance between each point and its centroid
# then add them up to assess the quality of the clusters
def distortion_cost(X,data_clusters,C):
    distortion=0 # intitialsing distortion value
    k=C.shape[0] # assigning a value to k to match the number of rows in the centroids array
    for i in range(k): # loops through rows in centroid array
        diffs=np.subtract(X[data_clusters==i],C[i]) # find differences between each centroid 
                                                    # and the data points in its cluster
        squareds=np.power(diffs,2) # squaring the difference
        sum_squared=np.sum(squareds) # summing up the total squared differences for each centroid
        distortion+=sum_squared # iteratively adding up the squared differences for all centroids 
    return distortion

######################################################################################
# Loops through the whole algorithm for each set number of different initial centroids
######################################################################################

for j in range(num_restarts):
    N = np.zeros(shape=(k,1)) # Initialise array to hold the number of rows in a cluster for a given restart
    
    C = create_centroids(X, k) # create k random centroids from data

    for i in range(t): # implementing the algorithm over t iterations
        M_indexes = np.random.choice(len(X), size=b, replace=False) # finding M distinct choices
        M = create_batch(X, M_indexes) # create batch
        m_batch_clusters = assign_centroids(M, C, k) # assigning centroids to points in the mini-batch
        C = gradient_step(m_batch_clusters, M, C, N) # updates centroids with gradient step
    
    # finding clusters for the whole dataset based on final 
    # set of centroids for the restart
    data_clusters = assign_centroids(X, C, k)     
    
    # Find value of cost function for the dataset for each restart
    distortions[0,j] = distortion_cost(X, data_clusters, C)         
    

distortions
np.min(distortions[0])
