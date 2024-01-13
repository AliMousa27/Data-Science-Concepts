from collections import Counter,defaultdict
import enum
from nlp import cosineSimilarity
from typing import List,Tuple,Dict

usersInterests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

def mostPopularNewInterest(userInterests :List[str],c: Counter, maxResults: int = 5) ->List[str]:
    suggestions = [(topic,freq) for topic,freq in c.most_common()
                   if topic not in userInterests]
    return suggestions[:maxResults]
    
def createUserInterestVector(user: List[str], uniqueInterests: List[str]):
    """
    Given a list ofinterests, produce a vector whose ith element is 1
    if unique_interests[i] is in the list, 0 otherwise
    """
    return [1 if interest in user else 0
            for interest in uniqueInterests]

#returns a list of similar users given a simialrity vector
def mostSimilarUsersTo(userId: int,similarityVector: List) -> List[Tuple[int,float]]:
    pairs = [(otherUser, similarity) for otherUser,similarity in enumerate(similarityVector[userId])
             if otherUser!= userId and similarity > 0]
    #key is similarity
    return sorted(pairs, key=lambda pair: pair[-1], reverse=True)

def userBasedSugestion(userId: int,similarityVector: List[List[str]], includeCurrentInterests: bool = False,):
    #dict where key is topic and val is similairty score
    suggestions : Dict[str,float] = defaultdict(float)
    #loop over each other user and how similar they are
    for otherUser,similarity in mostSimilarUsersTo(userId,similarityVector):
        #loop over each topic of the given user
        for topic in usersInterests[otherUser]:
            #add similarity score to that topic
            #print(suggestions[topic])
            #print(f"similarity iks {similarity}")
            suggestions[topic]+=similarity
            
    suggestions = sorted(suggestions.items(), key= lambda tuple: tuple[-1], reverse=True) # key is the similarity
    if includeCurrentInterests:
        return suggestions
    else:
        return [(suggestion,weight) for suggestion,weight in suggestions
                if suggestion not in usersInterests[userId]]
def main():
    counter = Counter(word
                      for user in usersInterests
                      for word in user)
    #print(mostPopularNewInterest(usersInterests[1],counter))
    uniqueInterests = sorted({topic for user in usersInterests for topic in user})
    interestVectors = [createUserInterestVector(user,uniqueInterests) for user in usersInterests]
    #userSimilarites[i][j] gives similarity score between user i and j
    userSimilarities = [[cosineSimilarity(userVecI,userVecJ) 
                        for userVecI in interestVectors]
                        for userVecJ in interestVectors
                        ]
    #print(mostSimilarUsersTo(0,userSimilarities))
    print(userBasedSugestion(0,userSimilarities))
if __name__ == "__main__": main()