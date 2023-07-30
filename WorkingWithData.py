import datetime
import math,random
from turtle import width
from typing import List,Dict
from collections import Counter, namedtuple
import matplotlib.pyplot as plt
import probabilty, vectors,statistics
from dateutil.parser import parse
import csv,re

#this function takes a point and classifies it in a bucket in order to make a histogram of it at a alter point
def bucketize(point: float, bucketSize:float) -> float:
    #divide by the size and floor it in order to classify data in buckets for the hiustorgram
    #example: point is 53 and bucketsize is 10
    #10*5=50 so we put it into 50 bucket
    return bucketSize * math.floor(point/bucketSize)
    
def createHistogramDict(dataPoints : List[float], bucketSize: float) -> Dict[float,int]:
    return Counter(bucketize(point,bucketSize) for point in dataPoints)    

def plotHistogram(points: List[float], bucketSize: float, title: str = ""):
    histogramVals = createHistogramDict(points,bucketSize)
    plt.bar(histogramVals.keys(),histogramVals.values(),width=bucketSize)
    plt.title(title)
    plt.show()
    
uniform = [200*random.random() for i in range(5000)]

#lotHistogram(points=uniform,10,"test")

#returns a random point from a normal disturbution of mean 0 and stdev 1
def randomNormalSample()->float:
    return probabilty.inverse_normal_cdf(random.random())

normalSample = [randomNormalSample() for i in range(5000)]
normalSample2 = [randomNormalSample() for i in range(5000)]
matrix=[normalSample,
        normalSample2]
#plotHistogram(points=normalSample,bucketSize=0.1,title="normal Test")

#to look at multi dimensional data we can try a correlation matrix, whos i,jth entry is the correlation between these 2 datapoints
#each row and column are a list of data themselves so we take the correlation between each datda set
def correleationMatrix(data:List[List]):
    def iAndjCorrelation(i:int,j:int)->float:
        return statistics.correlation(data[i],data[j])
    return vectors.make_matrix(len(data),len(data),iAndjCorrelation)

#print(correleationMatrix(matrix))

heights = [168, 175, 160, 182, 155, 190, 165, 178, 172, 159, 185, 173, 161, 188, 166, 177, 158, 170, 181, 176]
weights = [65, 70, 58, 80, 52, 90, 62, 75, 68, 56, 78, 70, 60, 85, 63, 72, 55, 67, 79, 74]

#returns stdev and mean for each positon
def scale(data: List[List[float]]):
    n= len(data[0])
    means = vectors.vectorMeans(data)
    stdevs = [statistics.stdev(vector[i]) for vector in data
                  for i in range(n)]
    return means,stdevs
    
data = [
    [168, 65],
    [175, 70],
    [160, 58],
    [182, 80],
    [155, 52],
    [190, 90],
    [165, 62],
    [178, 75],
    [172, 68],
    [159, 56],
    [185, 78],
    [173, 70],
    [161, 60],
    [188, 85],
    [166, 63],
    [177, 72],
    [158, 55],
    [170, 67],
    [181, 79],
    [176, 74]
]
