import re
from bs4 import BeautifulSoup 
import requests
from collections import defaultdict
from typing import Dict
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
    print(generateBigrams(transitions))
    
    trigrams = defaultdict(list)
    starts =[]
    for prev,current,next in zip(document,document[1:],document[2:]):
        if prev==".":
            starts.append(current)
        trigrams[(prev,current)].append(next)
        
        
    
    
if __name__ == "__main__" : main()