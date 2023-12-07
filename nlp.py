from matplotlib import pyplot as plt
import re
from bs4 import BeautifulSoup 
import requests
data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]
def fixUniCode(text : str) -> str:
    #u prefix denotes unicode string. This function replaces the unicode char with normal '
    return text.replace(u"\u2019","'")



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
        document.append(words)
    print(document)
if __name__ == "__main__" : main()