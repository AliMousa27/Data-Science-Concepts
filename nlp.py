import re
from tracemalloc import start
from bs4 import BeautifulSoup
from cv2 import exp 
import requests
from collections import defaultdict
from typing import Dict,List
import random

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
    "_N" : ["book", "computer", "friend", "movie", "car", "phone", "coffee", "music", "job", "house",
      "city", "food", "money", "time", "dog", "cat", "family", "work", "school", "health",
      "game", "travel", "party", "weather", "news", "internet", "camera", "friendship", "dream", "goal",
      "experience", "hobby", "exercise", "restaurant", "mind", "emotion", "relationship", "happiness", "art", "nature",
      "child", "parent", "holiday", "memory", "conversation", "skill", "challenge", "entertainment", "knowledge", "success",
      "failure", "achievement", "problem", "solution", "idea", "future", "past", "present", "community", "culture",
      "habit", "joy", "sadness", "surprise", "anger", "fear", "anxiety", "peace", "love", "freedom",
      "technology", "innovation", "communication", "creativity", "change", "progress", "simplicity", "complexity", "tradition", "discovery",
      "imagination", "curiosity", "passion", "challenge", "laughter", "play", "friendship", "relationship", "conversation", "cooperation",
      "support", "growth", "learning", "discovery", "reflection", "inspiration", "motivation", "resilience", "gratitude", "forgiveness"],
"_A" : ["happy", "sad", "funny", "serious", "tasty", "fast", "slow", "loud", "quiet", "big",
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
"_P" : ["about", "near", "over", "under", "behind", "in", "on", "off", "between", "through",
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
    print(generateSentence())
    
if __name__ == "__main__" : main()