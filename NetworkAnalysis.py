from typing import NamedTuple,Dict, List,Tuple
from vectors import Matrix, make_matrix,shape, Vector,dot,magnitude,distance
from collections  import deque
import random
from pprint import pprint

Path = List[int]
Friendships = Dict[int, List[int]]
class User(NamedTuple):
    id: int
    name: str
users = [User(0, "Hero"), User(1, "Dunn"), User(2, "Sue"), User(3, "Chi"),
         User(4, "Thor"), User(5, "Clive"), User(6, "Hicks"),
         User(7, "Devin"), User(8, "Kate"), User(9, "Klein")]

friendPairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#scuffed breadth first search
def shortestPathsFrom(fromUserId: int,
                        friendships: Friendships) -> Dict[int, List[Path]]:
    # A dictionary from "userId" to *all* shortest paths to that user
    shortestPathsTo: Dict[int, List[Path]] = {fromUserId: [[]]}
    #que that stores tuples of the from user and their neighbor
    frontier = deque((fromUserId,friendID) for friendID in friendships[fromUserId])
    while frontier:
        prevUserId, userId = frontier.popleft()
        
        pathsToPrev = shortestPathsTo[prevUserId]
        #add the new user to the path
        newPathToUser = [path + [userId] for path in pathsToPrev]
        oldPathsToUser = shortestPathsTo.get(userId,[])
        if oldPathsToUser:
            minPathLength = len(oldPathsToUser[0])
        else:
            minPathLength = float('inf')
            
        newPathToUser = [path for path in newPathToUser if len(path)<= minPathLength 
                         and path not in oldPathsToUser]
        
        shortestPathsTo[userId] = oldPathsToUser+newPathToUser
            # Add never-seen neighbors to the frontier
        frontier.extend((userId, friendId)
                        for friendId in friendships[userId]
                        if friendId not in shortestPathsTo)
    return shortestPathsTo
def farness(userId: int, shortestPaths):
    """the sum of the lengths of the shortest paths to each other user"""
    return sum(len(paths[0])
               for paths in shortestPaths[userId].values())
    
def makeTimesMatrix(m1:Matrix, m2: Matrix) -> Matrix:
    #number of rows and columns
    nr1,nc1 = shape(m1)
    nr2,nc2 = shape(m2)
    assert nc1 == nr2
    def entryFN(i,j) -> float:
        return sum(m1[i][k] * m2[j][k] for k in range(nc1))
    return make_matrix(nr1,nc2,entryFN)

def matrixTimesVector(m: Matrix, v: Vector):
    nr,nc = shape(m)
    assert nc == len(v)
    return [dot(v,row) for row in m]

def findEigenVector(A:Matrix,tolerance: float = 0.00001) -> Tuple[Vector,float]:
    guess = [random.random() for _ in A]
    while True:
        result = matrixTimesVector(A,guess)
        length = magnitude(result)
        #scale magnitude to be 1
        nextGuess = [x/length for x in result]
        if distance(nextGuess,guess) <tolerance:
            return nextGuess,length
        guess=nextGuess
def main():
    #type
    friendships : Friendships = {user.id: [] for user in users}
    for i,j in friendPairs:
        friendships[i].append(j)
        friendships[j].append(i)
    shortestPaths = {user.id: shortestPathsFrom(user.id,friendships) for user in users}
    betweenCentrality = {user.id : 0.0 for user in users}
    for source in users:
        for targetId, paths in shortestPaths[source.id].items():
            if source.id < targetId:      # don't double count
                num_paths = len(paths)     # how many shortest paths?
                contrib = 1 / num_paths    # contribution to centrality
                for path in paths:
                    for between_id in path:
                        if between_id not in [source.id, targetId]:
                            betweenCentrality[between_id] += contrib
    #print(betweenCentrality)
    closenessCentrality = {user.id: 1 / farness(user.id,shortestPaths) for user in users}
    #sprint(closenessCentrality)
    def entry_fn(i: int, j: int):
        return 1 if (i, j) in friendPairs or (j, i) in friendPairs else 0

    n = len(users)
    adjacency_matrix = make_matrix(n, n, entry_fn)
    #pprint(adjacency_matrix)
    eigenVector = findEigenVector(adjacency_matrix)
    pprint(eigenVector)

if __name__ == "__main__": main() 