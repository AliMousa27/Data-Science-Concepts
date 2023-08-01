from typing import List
from collections import Counter

data=["a","b","b","c","a"]
def majorityVote(labels: List[str]):
    """Assumes that labels are ordered from nearest to farthest."""
    voteCounts = Counter(labels)
    winner, winnerCount = voteCounts.most_common(1)[0]
    #find if we have the same amount of count for more than 1 item
    numOfWinners= len([val for val in  voteCounts.values() if val==winnerCount])
    if numOfWinners==1:
        return numOfWinners
    else:
        #try again without the farthest aka the last element to find a winner that ios truely closest to K
        return majorityVote(labels[:-1])
        
        
    
majorityVote(data)