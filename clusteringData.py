# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:15:40 2023

@author: jverl
"""
import numpy as np

data = np.genfromtxt("clusteringData.csv", delimiter=',')

data.shape

# Checking of scaling is needed
np.min(data, axis=0) 
np.max(data, axis=0)
# the min and max values for each feature seem to be from around -25 to 23
# Rescaling doesn't seem neccesary  

# Checking for missing values
np.isnan(data).sum() # No data is missing

# Checking for duplicate data
np.unique(data).shape # there are 640,000 unique rows compared to 800,000 in total
# However, there is no reason to think that 2 (or more) valid rows couldn't be the same 


# Creating initial centroids
def create_centroids(data, k):
    initial_index = np.random.randint(low=0, high=len(data)-1)# chooses a random row index
    initial_centroid = data[initial_index] # finds selected row and saves it as initial centroid 
    #current_centroid
    centroids = np.zeros(shape=(k,data.shape[1])) # create array of zeros to hold the centroids
    centroids[0,]=initial_centroid # setting the first centroid as initial_centroid
    #centroids
    for i in range(1,k,1):
        diffs = np.subtract(data,initial_centroid)  # Finding the difference between each point and a centroid
                                                    # first loop will be the initial_centroid, then, the next...  
        diffs_squared = np.power(diffs, 2)  # squares the differences
        sum_of_squared_diffs_by_row = np.sum(diffs_squared, axis=1) # sums up the squared differences forv each row 
        sum_of_squared_diffs_total = np.sum(diffs_squared) # Finding the total sum of squared differences of the whole array together
        probs = np.divide(sum_of_squared_diffs_by_row,sum_of_squared_diffs_total) # array of probabilities proportional to euclidian distance 
                                                                                  # from the current centroid
        current_centroid_index = np.random.choice(len(data),1,p=probs) # Finding index for next centroid with 
                                                                       # probability based on previous line
        current_centroid = data[current_centroid_index,] # Finding next centroid
        centroids[i,] = current_centroid # saving next centroid to centroid array
    return centroids

#centroids=create_centroids(data, 3)
#centroids
def calculate_distance(data,point):
    diffs=np.subtract(data,point) # finds the difference by element between each 
                                  # row of data and a single query point 
    absolute_diffs=np.absolute(diffs) # finds the absolute value
    manhattens=np.sum(absolute_diffs, axis=1) # sums up absolutes to find the Manhatten Distance metric
    return manhattens
    
def assign_centroid(data, centroids):
    k=centroids.shape[0] # assigning a value to k to match the number of rows in the centroids array
    distance_array=np.zeros(shape=(k,len(data))) # creating array to hold manhatten distances 
                                                 # between each centroid and all other rows in data

    for i in range(k):
        distance_array[i]=calculate_distance(data, centroids[i]) # loops through each row to find distances (which are stored by column)
    
    centroid_indices=np.argmin(distance_array,axis=0) # Finds the index of each row that has the minimum distance from that rows centroid
    return centroid_indices

#cluster_indices=assign_centroid(data, centroids)
#cluster_indices.shape

def move_centroids(data,cluster_indices,centroids):
    k=centroids.shape[0] # # assigning a value to k to match the number of rows in the centroids array
    for i in range(k):
        centroids[i]=data[cluster_indices==i].mean(axis=0) # boolean index finds all the rows for a 
                                                           # cluster/centroid i, then calculates the 
                                                           # mean by column and saves it as the new centroid
    return centroids
        
#move_centroids(data, cluster_indices, centroids)

def distortion_cost(data,cluster_indices,centroids):
    distortion=0 # intitialsing distortion value
    k=centroids.shape[0] # assigning a value to k to match the number of rows in the centroids array
    for i in range(k): # loops through rows in centroid array
        diffs=np.subtract(data[cluster_indices==i],centroids[i]) # find differences between each centroid 
                                                                 # and the data points in its cluster
        squareds=np.power(diffs,2) # squaring the difference
        sum_squared=np.sum(squareds) # summing up the total squared differences for each centroid
        distortion+=sum_squared # iteratively adding up the squared differences for all centroids 
    return distortion


num_iterations=10 # setting number of iteration per initial centroid
num_restarts=10 # setting number of restarts (with different initial centroids)

#k=3 # setting the value of k (number of clusters)
distortion_values=np.zeros(shape=(num_iterations,)) # creating a list to hold all 
                                                    # the minimum distortion costs for each start/restart

def restart_KMeans(data, k, num_iterations, num_restarts):
    clusters_restart=np.zeros(shape=(num_restarts,len(data))) # array of zeros to hold the clusters
                                                              # for the chosen iteration in each restart
    distortions_restart=np.zeros(shape=(num_restarts,1)) # array of zeros to hold the distortion costs
                                                         # for the chosen iteration in each restart
    for j in range(num_restarts): # outer loop for each restart
        distortion_values=np.zeros(shape=(num_iterations,)) # zero array to hold distortion costs for all iterations in this restart
        centroids=create_centroids(data, k) # creating the initial centroids for this restart

        cluster_index_list=np.zeros(shape=((num_iterations,len(data))))# zero array to hold the cluster 
                                                                       # indices for each iteration in this restart
        cluster_index_list
        for i in range(num_iterations): # inner loop to run each iteration for the initial centroids chosen
            cluster_indices=assign_centroid(data, centroids) # returns cluster indices for centroids
            cluster_index_list[i]=cluster_indices # saves each array of cluster indices to an array to the ith of cluster_index_list 
            distortion_values[i]=distortion_cost(data, cluster_indices, centroids) # finds distortion cost for this 
                                                                                   # iteration (i) and saves to array to 
                                                                                   # hold all of them
            centroids=move_centroids(data, cluster_indices, centroids) # # assigns new centroids based on algorithm
            

        optimal_cluster_index=np.argmin(distortion_values) # finds index of minimum distortion cost for all iterations in a restart
        optimal_clusters=cluster_index_list[optimal_cluster_index] # finds the corresponding cluster indices
        optimal_distortion_value=distortion_values[optimal_cluster_index] # finds the corresponding distortion value
        clusters_restart[j]=optimal_clusters # saving the best cluster indices for this restarts 
        distortions_restart[j]=optimal_distortion_value # saving the best distortion cost for this restarts

    index_optimal_cluster=np.argmin(distortions_restart) # finds index of minimum distortion cost for all restarts
    final_clusters=clusters_restart[index_optimal_cluster] # finds corresponding cluster
    best_distortion=distortions_restart[index_optimal_cluster] # Finds corresponding distortion cost
    output=np.hstack((final_clusters,best_distortion)) # creates final output with optimal 
                                                       # clusters and the distortion cost ascociated with it
    return output

output=restart_KMeans(data, 6, num_iterations, num_restarts) # k=6 was chosen as best k
output.shape

final_distortion=output[len(output)-1] # final distortion value
final_clusters=output[0:(len(output)-2)] # final clusters
final_distortion
final_clusters
#len(output)
#len(final_clusters)

num_outputs = 10 # setting the maximum k value
outputs=np.zeros(shape=(num_outputs,len(data)+1)) # empty array to hold different outputs for varying k

for i in range(num_outputs): # loops through k values from 1 to 10
    print(i)
    output=restart_KMeans(data, i+1, num_iterations, num_restarts) # finding output for k
    outputs[i]=output # passing value to outputs

import matplotlib.pyplot as plt # importing matplotlib

y = outputs[:,len(data)] # taking distortion values from the outputs
#len(y)
x = range(1,num_outputs+1) # range of k values
#len(x)

plt.plot(x, y) # plotting x against y
plt.show()    
