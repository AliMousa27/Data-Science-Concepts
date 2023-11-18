from os import name
import random
import numpy as np
from scipy import cluster
import tqdm
from vectors import Vector,vector_mean,squared_distance, distance
from typing import List, NamedTuple, Union, Callable
import itertools
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


#compute the number of differnece between 2 vectors
def numberOfDifferences(v1: Vector, v2: Vector) -> Vector:
    assert len(v1)==len(v2)
    return len([x1 for x1,x2 in zip(v1,v2) if x1!=x2])

#given a vector and their assigned clusters compute mean of cluster
def clusterMeans(k: int,
                  inputs: List[Vector],
                  assignments: List[int]) -> List[Vector]:
    #create k clusters
    clusters = [[] for _ in range(k)]
    for input,assignment in zip(inputs,assignments):
        # clusters[i] contains the inputs whose assignment is i
        #add the input to its assigned cluster
        clusters[assignment].append(input)
        #chose random input for empty cluster to be the mean
    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]

        
class KMeans():
    def __init__(self, k:int) -> None:
        self.means = None
        self.k=k
        
    #returns the assigned the cluster to each input
    def classify(self,input: Vector)->int:
        #make an iterable of k-1 vals then squared distance between the input and each cluster
        #and return the index of the min cluster
        return min(range(self.k),
                   key=lambda i: squared_distance(self.means[i], input))
        
    def train(self,inputs:List[Vector]):
        #start with random assignment for each input vector
        assignments = [random.randrange(self.k) for _ in inputs]
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                #compute means
                self.means = clusterMeans(self.k,inputs,assignments)
                #find new assignments
                newAssignments = [self.classify(input) for input in inputs]
                numOfChanges = numberOfDifferences(assignments,newAssignments)
                if numOfChanges == 0:
                    return
                else:
                    self.means = clusterMeans(self.k,inputs,newAssignments)
                    assignments=newAssignments

def squaredClusteringErrors(k: int, inputs: List[Vector]) -> float:
    clusters = KMeans(k)
    clusters.train(inputs)
    assignments = [clusters.classify(input) for input in inputs]
    #sum of th total squared distance between each input and its assigned cluster
    return sum(squared_distance(clusters.means[assignment], input) 
               for input,assignment in zip(inputs,assignments))
#method 2: bottom up clustering, merge each input with the one closest to it
class Leaf(NamedTuple):
    value: Vector
leaf1= Leaf([10,20])
leaf2= Leaf([30,-15])
class Merged(NamedTuple):
    children: tuple
    order: int
#a merged cluster with 2 merged leaves means its of second order
merged = Merged(children=(leaf1,leaf2),order=2)
#a given cluster cna be either a leaf or merged
Cluster = Union[Leaf,Merged]


#helper function to get all the values inside a cluster
def getValues(cluster: Cluster):
    #base case
    if isinstance(cluster,Leaf):
        return [cluster.value],
    else:
        #recursivly go through each child in cluster and get its values as a list
        return [value for child in cluster.children for value in getValues(child)]
    
def clusterDistance(cluster1: Cluster,cluster2: Cluster,
                    #stratget could be for example the minimum distance between clusters
                    strategy:Callable = min):
    #get all distances between clus1 and clus2 then get min distance 
    return strategy(
        [distance(
            [v1,v2] for v1 in cluster1
            for v2 in cluster2
        )]
    )
def getMergeOrder(cluster: Cluster):
    if isinstance(cluster,Leaf):
        #we havent merged yet
        return float('inf')
    return cluster.order
def getChildren(cluster:Cluster):
    if isinstance(cluster,Leaf):
        #we havent merged yet
        raise TypeError("Leaf aint got no children")
    return cluster.children

def bottomUpCluster(inputs: List[Vector],
                    strategy: Callable=min):
    #start with leaves
    clusters: List[Cluster]= [Leaf(input) for input in inputs]
    def pairDistance(pair: tuple(Cluster,Cluster)):
        return clusterDistance[pair[0],pair[1]]
    #while  we still have clusters to merge
    while len(clusters)>1:
        #find 2 closest clusters
        c1,c2 = min(((cluster1,cluster2) for i,cluster1 in enumerate(clusters)
                    for cluster2 in cluster1[:i]),key=pairDistance)    
        #remove them from the cluster
        clusters = [cluster for cluster in clusters if c1!=cluster and c2!=cluster]
        #merge the clusters
        mergedClusters = Merged((c1,c2), order=len(clusters))
        #add the merged to the clusters
        clusters.append(mergedClusters)
    return clusters[0]    
    return clusters[0]

def generate_clusters(base_cluster: Cluster,
                      num_clusters: int) -> List[Cluster]:
    # start with a list with just the base cluster
    clusters = [base_cluster]

    # as long as we don't have enough clusters yet...
    while len(clusters) < num_clusters:
        # choose the last-merged of our clusters
        next_cluster = min(clusters, key=getMergeOrder())
        # remove it from the list
        clusters = [c for c in clusters if c != next_cluster]

        # and add its children to the list (i.e., unmerge it)
        clusters.extend(getChildren(next_cluster))

    # once we have enough clusters...
    return clusters
def main():
    '''inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    random.seed(12)                   
    clusterer = KMeans(k=3)
    clusterer.train(inputs)
    means = sorted(clusterer.means)   
        
    assert len(means) == 3
    # Check that the means are close to what we expect.
    assert squared_distance(means[0], [-44, 5]) < 1
    assert squared_distance(means[1], [-16, -10]) < 1
    assert squared_distance(means[2], [18, 20]) < 1
        
    random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    means = sorted(clusterer.means)
        
    assert len(means) == 2
    assert squared_distance(means[0], [-26, -5]) < 1
    assert squared_distance(means[1], [18, 20]) < 1'''
    
    '''ks = range(1, len(inputs) + 1)
    errors = [squaredClusteringErrors( k,inputs) for k in ks]
    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.title("Total Error vs. # of Clusters")
    plt.show()'''
    
    '''  jogoatPath = r"jogoat.png"
    jogoat = mpimg.imread(jogoatPath)
    pixels = [pixel.tolist() for row in jogoat for pixel in row]
    #create 5 clusters aka the 5 most prevelant colors
    kmeans = KMeans(k=2)
    kmeans.train(pixels)
    #recolor the pixel
    def recolor(pixel:Vector):
        cluster = kmeans.classify(pixel)
        return kmeans.means[cluster]
    newJogoat = [[recolor(pixel) for pixel in row] for row in jogoat]
    plt.imshow(newJogoat)
    plt.axis("off")
    plt.show()'''
    
        
if __name__ == "__main__":main()