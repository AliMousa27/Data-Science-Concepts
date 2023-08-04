#function to extract all the words and remove duplicates
import re
import token
from typing import NamedTuple, List, Tuple,Iterable,Dict,Set
from collections import defaultdict
#class for the training data
class Message(NamedTuple):
    text:str
    isSpam:bool

def tokenize(text: str)-> set[str]:
    text=text.lower()
    allWords = text.split(" ")
    return set(allWords)

#creating a class sthat keeps track of the counts and labels of the training data
class NaiveBayesClassifier:
    #k is the psuedo count to not have unrealisitc probabilities in the data set
    def __init__(self,k:float = 0.5) -> None:
        self.tokens=Set[str] = set()
        #dict where key is word and val is how many times the word appeared in a spam email
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        #ham is not spam email
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        #initalize spam and hams to be 0 initally
        self.spam_messages = self.ham_messages = 0

    #the model gets an iterable of messages. 
    #1. increment spam or ham message based on the bool
    #2. add them to the dict based on if they are spam or ham
    def train(self,messages: Iterable[Message]) -> None:
        for message in messages:
            if message.isSpam:
                self.spam_messages+=1
            else:
                self.ham_messages+=1
                #then for each of the words we increment the dict val counts based on if its spam or ham
            for token in tokenize(message):
                if message.isSpam:
                    self.token_spam_counts[token]+=1
                else:
                    self.token_ham_counts[token]+=1