import json
import math
import re
from bs4 import BeautifulSoup
from numpy import dot
import requests
from collections import defaultdict
from typing import Counter, Dict,List, Tuple
import random
from DeepLearning import Tensor, randomTensor, zeroesTensor
import tqdm
from typing import Iterable
from DeepLearning import Layer, Tensor, random_tensor, zeros_like
from vectors import Vector

def fixUniCode(text : str) -> str:
    #u prefix denotes unicode string. This function replaces the unicode char with normal '
    return text.replace(u"\u2019","'")

#transitions a dict where the key is a word and value is lsit of words that follow it
def generateBigrams(transitions : Dict) -> str :
    current = "."#means that next word will start a sentence
    
    result = []
    while True:
        #probable words that should follow
        candidates=transitions[current]
        current = random.choice(candidates)
        result.append(current)
        if current == "." : return " ".join(result)
Grammar = Dict[str, List[str]]
#symbols starting with underscores are non terminal symbols
grammar = {
    "_S"  : ["_NP _VP"],#starting symbol
    "_NP" : ["_N",#noun phrase
             "_A _NP _P _A _N"],
    "_VP" : ["_V",#verb phrase
             "_V _NP"],
    "_N" : ["book","zepei", "computer", "friend", "movie", "car", "phone", "coffee", "music", "job", "house",
      "city", "food", "money", "time", "dog", "cat", "family", "work", "school", "health",
      "game", "travel", "party", "weather", "news", "internet", "camera", "friendship", "dream", "goal",
      "experience", "hobby", "exercise", "restaurant", "mind", "emotion", "relationship", "happiness", "art", "nature",
      "child", "parent", "holiday", "memory", "conversation", "skill", "challenge", "entertainment", "knowledge", "success",
      "failure", "achievement", "problem", "solution", "idea", "future", "past", "present", "community", "culture",
      "habit", "joy", "sadness", "surprise", "anger", "fear", "anxiety", "peace", "love", "freedom",
      "technology", "innovation", "communication", "creativity", "change", "progress", "simplicity", "complexity", "tradition", "discovery",
      "imagination", "curiosity", "passion", "challenge", "laughter", "play", "friendship", "relationship", "conversation", "cooperation",
      "support", "growth", "learning", "discovery", "reflection", "inspiration", "motivation", "resilience", "gratitude", "forgiveness"],
"_A" : ["happy", "gay","sad", "funny", "serious", "tasty", "fast", "slow", "loud", "quiet", "big",
      "small", "hot", "cold", "bright", "dark", "hard", "soft", "new", "old", "young",
      "fast", "slow", "light", "heavy", "clean", "messy", "simple", "complex", "healthy", "unhealthy",
      "easy", "difficult", "safe", "dangerous", "rich", "poor", "beautiful", "ugly", "interesting", "boring",
      "creative", "ordinary", "active", "lazy", "modern", "traditional", "flexible", "rigid", "exciting", "calm",
      "crazy", "normal", "important", "trivial", "silly", "wise", "generous", "selfish", "brave", "fearful",
      "patient", "impatient", "positive", "negative", "hopeful", "hopeless", "productive", "unproductive", "organized", "disorganized",
      "motivated", "lazy", "ambitious", "content", "proud", "embarrassed", "sincere", "insincere", "honest", "dishonest",
      "friendly", "unfriendly", "helpful", "unhelpful", "tolerant", "intolerant", "polite", "rude", "flexible", "stubborn",
      "curious", "indifferent", "passionate", "apathetic", "caring", "uncaring", "confident", "insecure", "assertive", "timid",
      "ethical", "unethical", "fair", "unfair", "responsible", "irresponsible", "adventurous", "cautious", "spontaneous", "predictable"],
"_P" : ["about","is", "near", "over", "under", "behind", "in", "on", "off", "between", "through",
      "across", "against", "around", "before", "after", "during", "along", "amidst", "beyond", "inside",
      "outside", "towards", "away", "within", "without", "throughout", "despite", "because of", "in spite of", "due to",
      "instead of", "next to", "opposite", "among", "beside", "between", "past", "against", "through", "over",
      "under", "behind", "in front of", "on top of", "below", "above", "with", "without", "beneath", "alongside",
      "amid", "after", "before", "during", "inside", "outside", "nearby", "far away", "overhead", "underfoot", "behind",
      "in the middle of", "on the edge of", "around the corner", "through the woods", "on the horizon", "across the street", "beyond the mountains", "within reach", "out of reach"],
"_V" : ["learns", "trains", "tests", "is", "reads", "writes", "listens", "speaks", "plays", "creates",
      "builds", "designs", "solves", "meets", "challenges", "achieves", "fails", "grows", "changes", "adapts",
      "learns", "teaches", "helps", "supports", "listens", "understands", "expresses", "shares", "communicates", "collaborates",
      "decides", "plans", "organizes", "manages", "facilitates", "enjoys", "appreciates", "values", "encourages", "motivates",
      "inspires", "influences", "impacts", "contributes", "volunteers", "participates", "cares", "apologizes", "forgives", "succeeds",
      "struggles", "celebrates", "reflects", "resolves", "negotiates", "compromises", "creates", "innovates", "explores", "discovers",
      "improves", "optimizes", "analyzes", "synthesizes", "visualizes", "implements", "evaluates", "measures", "tests", "validates",
      "optimizes", "iterates", "designs", "builds", "debugs", "fixes", "maintains", "upgrades", "customizes", "integrates",
      "debugs", "customizes", "integrates", "deploys", "manages", "monitors", "troubleshoots", "protects", "secures", "ensures",
      "optimizes", "improves", "innovates", "creates", "designs", "facilitates", "organizes", "plans", "solves", "implements"]}
documents = [
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

k=4
#the following will be lists to help calculate the sample weights for topic modeling method to calcualte the sample weights
#initlaize an empty set of counters for each document to count how many times said words appear
#will keep track how many times each word appears in each doc
documentTopicCounts = [Counter() for _ in documents]
#how many times each word is assigned to each topic
topicWordCounts = [Counter() for _ in range(k)]
#total number of words for each topic
topicCount = [0 for _ in range(k)]
#total number of words contained in each doc
documentLengths = [len(doc) for doc in documents]
#number of distinct words
distinctWords = set(word for doc in documents for word in doc)
W = len(distinctWords)
#number of documents
D = len(documents)
#once we populate these counter lists we can find the the number of words in doc[3] that are assoicated in topic 1
#like so docTopicCounts[3][1]
#and find the number of times nlp is assocated with topic 1 like so topicCount[1]["nlp"]
def pTopicGivenDocument(topic: int, d: int, alpha: float = 0.1) -> float:
    #fraction of words in doc d assoctaed with the topic / total length of said document
    return ((documentTopicCounts[d][topic]+alpha)/
            (documentLengths[d]+k*alpha))
def pWordGivenTopic(word: str, topic: int, beta: float = 0.1) -> float:
    #fraction of how often a word appears in a topic / topic length
    return ((topicWordCounts[topic][word] + beta)/
            (topicCount[topic] + W * beta))
def topicWeight(d: int, word: str, k: int) -> float:
    """
    Given a document and a word in that document,
    return the weight for the kth topic
    """
    return pTopicGivenDocument(k,d) * pWordGivenTopic(word,k)

def chooseNewTopic(d: int, word: str) -> int:
    return sampleFrom([topicWeight(d, word, k)
                        for k in range(k)])
    
def isTerminal(token : str) -> bool:
    return token[0] !="_"
def generateSentence():
    return expand(grammar,["_S"])   
def expand(grammar:Grammar, tokens: List[str]) -> list[str]:
    for i,token in enumerate(tokens):
        #skip token
        if isTerminal(token) : continue
        symbol : str = random.choice(grammar[token])
        #replace the non term with the term
        if isTerminal(symbol):
            tokens[i] = symbol
        else:
            #symbol is non term so could  be _np _vp
            #so expand the tokens set by getting from start to i, expand non term then get the rest
            tokens = tokens[:i] + symbol.split() + tokens[(i+1):]
        #recurisly call it on the new token list
        return expand(grammar,tokens)
    return tokens
         
         
def rollADice() -> int:
    return random.choice([1, 2, 3, 4, 5, 6])

def directSample() -> Tuple[int, int]:
    d1 = rollADice()
    d2 = rollADice()
    return d1, d1 + d2

def randomYGivenX(x: int) -> int:
    """equally likely to be x + 1, x + 2, ... , x + 6"""
    return x + rollADice()

def randomXgivenY(y: int) -> int:
    if y <= 7:
        # if the total is 7 or less, the first die is equally likely to be
        # 1, 2, ..., (total - 1)
        return random.randrange(1, y)
    else:
        # if the total is 7 or more, the first die is equally likely to be
        # (total - 6), (total - 5), ..., 6
        return random.randrange(y - 6, 7)

def gibbsSample(num_iters: int = 100) -> Tuple[int, int]:
    x, y = 1, 2 # doesn't really matter
    for _ in range(num_iters):
        x = randomXgivenY(y)
        y = randomYGivenX(x)
    return x, y

def compareDistributions(num_samples: int = 1000) -> Dict[int, List[int]]:
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbsSample()][0] += 1
        counts[directSample()][1] += 1
    return counts

#randomly choose an index based on weights
def sampleFrom(weights: List[float]) -> int:
    """returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total*random.random()
    for i, w in enumerate(weights):
        rnd -= w                       # return the smallest i such that
        if rnd <= 0: return i          # weights[0] + ... + weights[i] >= rnd
def cosineSimilarity(v1: Vector, v2: Vector) -> float:
    return dot(v1,v2)/math.sqrt(dot(v1,v1)*dot(v2,v2))

colors = ["red", "green", "blue", "yellow", "black", ""]
nouns = ["bed", "car", "boat", "cat"]
verbs = ["is", "was", "seems"]
adverbs = ["very", "quite", "extremely", ""]
adjectives = ["slow", "fast", "soft", "hard"]

def makeSentence() -> str:
    return " ".join([
        "The",
        random.choice(colors),
        random.choice(nouns),
        random.choice(verbs),
        random.choice(adverbs),
        random.choice(adjectives),
        "."
    ])
    
    
#Class to keep track of word and its word id
class Vocabulary:
    def __init__(self, words: List[str] = None) -> None:
        #w2i = word to id
        self.w2i : Dict[str,int] = {} # maps word to word id
        self.i2w = Dict[int,str] = {} # maps id to word
        #if words were provided, using or operator as a fallback mechanism if words are empty
        for word in (words or []): 
            self.add(word)
    #Property is used to mark functions that should be treated as attributes
    @property
    def size(self) -> int:
        #how many words in the vocab
        return len(self.i2w)
    
    def add(self, word: str) -> None:
        if word not in self.w2i:
            wordID = len(self.w2i)
            #add to both dicts
            self.w2i[word] = wordID
            self.i2w[wordID] = word
    def getWord (self, i: int) -> str:
        #get word from id
        return self.i2w[i]
    def getId(self, word: str) -> int:
        #get id from word
        return self.w2i[word]
    def oneHotEncode (self, word:str) -> Tensor:
        wordId = self.getId(word)
        assert wordId is not None,f"unknown word: {word}"
        return [1.0 if i == wordId else 0.0 for i in range(self.size)] 
    
def saveVocab(vocab: Vocabulary) -> None:
    with open("vocabFile","w") as f:
        json.dump(vocab.w2i,f)
def loadVocab(fileName:str) ->Vocabulary:
    vocab = Vocabulary()
    with open(fileName, "r") as f:
        vocab.w2i = json.load(f)
        vocab.i2w = {id:word for word, id in vocab.w2i.items()}
    return vocab

class Embedding:
    
    def __init__(self, numEmbeddings: int, embeddingDim: int) -> None:
        """
        Args:
            numEmbeddings (int): The number of embeddings.
            embeddingDim (int): The dimension of the embeddings.
        """
    def __init__(self,numEmbeddings:int, embeddingDim:int) -> None:
        self.numEmbeddings = numEmbeddings
        self.embeddingDim=embeddingDim
        #create vector of size embedding dim for however many embeddings we need
        self.embeddings = randomTensor(numEmbeddings,embeddingDim)
        self.grad=(zeroesTensor(self.embeddings))
        #save last input id
        self.lastInputID = None
        
    def forward(self, inputID: int):
        self.inputID = inputID #save for backpropagation
        return self.embeddings[inputID]
    
    
    #the gradient is zero for every embedding except for the chosen one
    def backward(self, gradient: Tensor) -> None:
        # Zero out the gradient corresponding to the last input.
        if self.lastInputID is not None:
            zeroList = [0 for _ in range(self.embeddingDim)]
            self.grad[self.lastInputID] = zeroList

        self.lastInputID = self.inputID
        self.grad[self.inputID] = gradient
    def params(self) -> Iterable[Tensor]:
        return [self.embeddings]
    def grads(self) -> Iterable[Tensor]:
        return [self.grad]

def main():
    url = "http://radar.oreilly.com/2010/06/what-is-data-science"
    page = requests.get(url).text
    soup = BeautifulSoup(page,"html5lib")
    content = soup.find("div", id="body-content")
    #one or more word characters. Since space isnt a word charcter then the first part
    #finds all the words bassically. The latter just finds apostrophes combine both and it becomes
    #A OR B
    regex = r"[\w]+|[\.]"
    document = []

    for paragraph in content.find_all("p"):
        words = re.findall(pattern=regex, string=fixUniCode(paragraph.text))
        document.extend(words)
    #create a deafult dict where the normal value for any key is an empty list
    transitions = defaultdict(list)
    #basically finding where each word ends and what word follows it
    for prev,current in zip(document,document[1:]):
        transitions[prev].append(current)    
    #will spew out gibbrish most likely
    #print(generateBigrams(transitions))
    
    trigrams = defaultdict(list)
    starts =[]
    for prev,current,next in zip(document,document[1:],document[2:]):
        if prev==".":
            starts.append(current)
        trigrams[(prev,current)].append(next)
    #print(generateTrigrams(starts=starts, trigrams=trigrams))
    #print(generateSentence())
    #assign every word to a random topic for now
    documentTopics = [[random.randrange(4) for word in doc] for doc in documents]
    for d in range(D):
        for word, topic in zip(documents[d],documentTopics[d]):
            documentTopicCounts[d][topic]+=1
            topicWordCounts[topic][word]+=1
            topicCount[topic]+=1
    for iter in tqdm.trange(1000):
        for d in range(D):
            for i, (word, topic) in enumerate(zip(documents[d],
                                                documentTopics[d])):

                # remove this word / topic from the counts
                # so that it doesn't influence the weights
                documentTopicCounts[d][topic] -= 1
                topicWordCounts[topic][word] -= 1
                topicCount[topic] -= 1
                documentLengths[d] -= 1

                # choose a new topic based on the weights
                new_topic = chooseNewTopic(d, word)
                documentTopics[d][i] = new_topic

                # and now add it back to the counts
                documentTopicCounts[d][new_topic] += 1
                topicWordCounts[new_topic][word] += 1
                topicCount[new_topic] += 1
                documentLengths[d] += 1
    for k, word_counts in enumerate(topicWordCounts):
        for word, count in word_counts.most_common():
            if count > 0:
                pass
                #print(k, word, count)
    sentences = [makeSentence() for _ in range(NUM_SENTENCES)]
    for sentence in sentences:
        print(" ".join(sentence))


NUM_SENTENCES = 50
    
if __name__ == "__main__" : main()