import math,random
from turtle import width
from typing import List,Dict
from collections import Counter
import matplotlib.pyplot as plt
import probabilty, vectors,statistics,linecache
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

normalSample = [randomNormalSample() for i in range(500000)]
#plotHistogram(points=normalSample,bucketSize=0.1,title="normal Test")

#to look at multi dimensional data we can try a correlation matrix, whos i,jth entry is the correlation between these 2 datapoints
#each row and column are a list of data themselves so we take the correlation between each datda set
def correleationMatrix(data:List[List]):
    def iAndjCorrelation(i:int,j:int)->float:
        return statistics.correlation(data[i],data[j])
    return vectors.make_matrix(len(data),len(data),iAndjCorrelation)

    