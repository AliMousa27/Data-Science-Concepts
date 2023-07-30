from collections import namedtuple,defaultdict
import csv
from dataclasses import dataclass
import datetime
from tokenize import maybe
from typing import NamedTuple,List
from dateutil.parser import parse
from numpy import percentile

@dataclass
class StockPrice:
    name: str
    date: datetime.date
    openPrice: float
    high: float
    low: float
    close: float
    adjClose: float
    volume: int
    
    def parseCSV(row:list[str]):
        try:
            name,date,openPrice,high,low,close,adjClose,volume=row
            return StockPrice(name=name,
                          date=parse(date).date(),
                          openPrice=float(openPrice),
                          high=float(high),
                          low=float(low),
                          close=float(close),
                          adjClose=float(adjClose),
                          volume=float(volume)
                          )
        except:
            print("couldnt parse")
    #comaprison methods to sort by date
    def __lt__(self, other):
        return self.date < other.date

    def __gt__(self, other):
        return self.date > other.date

    def __eq__(self, other):
        return self.date == other.date



@dataclass
class StockPricePercentChange:
    name: str
    date: datetime.date
    percent:float


def calcPercentIncrease(priceToday : float,priceYesterday:float) -> float: 
    return ((priceToday-priceYesterday)/priceYesterday)*100
    
allStocks: List[StockPrice] = []

with open("Stocks.csv","r") as csvFile:
    csvReader = csv.reader(csvFile)
    header = next(csvReader)
    for row in csvReader:
        # 'row' is a list of elements (fields) for each row
        maybeValid = StockPrice.parseCSV(row)
        if maybeValid is None:
            print(f"skipping row: {row}")
        else: allStocks.append(maybeValid)
        
priceByName: defaultdict[str, List[StockPrice]] = defaultdict(list)
#making a dict with name as the key and stock prices as its list of stocks
for stock in allStocks:
    priceByName[stock.name].append(stock)
#sort by date as the methods implemnented by the data class
priceByName = {symbol : sorted(stockPrice) for symbol, stockPrice in priceByName.items()}

#key is symbol value is percnt change data class
percentChangeWithMonths: defaultdict[str, List[StockPricePercentChange]] = defaultdict(list)

for yesterday, today in zip(allStocks, allStocks[1:]):
    name=today.name
    date=today.date
    percent=calcPercentIncrease(today.close,yesterday.close)
    percentChangeWithMonths[today.name].append(StockPricePercentChange(name=name,date=date,percent=percent))


applPercents = [(appl.percent,appl.date) for appl in percentChangeWithMonths["GOOG"]]
print(max(applPercents))