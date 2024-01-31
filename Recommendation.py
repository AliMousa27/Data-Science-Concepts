from collections import Counter,defaultdict
from nlp import cosineSimilarity
from typing import List,Tuple,Dict, NamedTuple
import csv
import re
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
        
        
def mostSimilarInterest(interestId: int,interestSimilarities:List,uniqueInterests: List):
    similarities = interestSimilarities[interestId]
    similarities = interestSimilarities[interestId]
    pairs = [(uniqueInterests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interestSimilarities != other_interest_id and similarity > 0]
    return sorted(pairs,
                  key=lambda pair: pair[-1],
                  reverse=True)
    
def itemBasedSuggestions(userId: int,userInterestsVector,interestSimilarities,uniqueInterests,
                           include_current_interests: bool = False):
    # Add up the similar interests
    suggestions = defaultdict(float)
    userInterest_vector = userInterestsVector[userId]
    for interestId, is_interested in enumerate(userInterestsVector):
        if is_interested == 1:
            similar_interests = mostSimilarInterest(interestId,interestSimilarities,uniqueInterests)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    # Sort them by weight
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in usersInterests[userId]]


class RATING(NamedTuple):
    userID : str
    movieID: str
    rating: float


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
    #print(userBasedSugestion(0,userSimilarities))
    
    #rows are interests columns are users
    interestMatrix = [[userInterestVector[j] for userInterestVector in interestVectors]
                          for j,_ in enumerate(uniqueInterests)]
    interestSimilarities = [[cosineSimilarity(v1,v2) for v1 in interestMatrix]
                            for v2 in interestMatrix]
    #doiwnoad file from https://files.group-lens.org/datasets/movielens/ml-100k.zip and put in appropriate directory
    MOVIES = "../ml-100k/u.item"
    RATINGS = "../ml-100k/u.data"
    
    with open(MOVIES, encoding="iso-8859-1") as f:
        reader = csv.reader(f,delimiter="|")
        #The * is for ignoring the other useless args split by the delimiter that we dont need
        
        movies = {movieID: title for movieID, title, *_ in reader}
        assert len(movies) == 1682
    with open(file=RATINGS, encoding="iso-8859-1") as f:
        reader = csv.reader(f,delimiter="\t")
        ratings = {RATING(userID,movieID,rating)
                 for userID,movieID,rating,_ in reader}
        assert len(list(rating.userID for rating in ratings)) == 100000
        
    #dict where movie id is the key and the value is a list of its ratings
    ratingsForBatman = {movieID: [] for movieID, title in movies.items() if re.search("Batman",title)}
    
    for rating in ratings:
        if rating.movieID in ratingsForBatman:
            ratingsForBatman[rating.movieID].append(rating.rating)
    print(ratingsForBatman)
    averageRating = {movies[movieID]: (sum(int(rating) for rating in ratings)/len(ratings)) for movieID, ratings in ratingsForBatman.items()}
    print(averageRating)
            
            
if __name__ == "__main__": main()