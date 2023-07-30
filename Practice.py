from collections import namedtuple,defaultdict
import csv
from dataclasses import dataclass
import datetime
from tokenize import maybe
from typing import NamedTuple,List
from dateutil.parser import parse

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
        
    
        
    
allStocks: List[StockPrice] = []
with open("test.csv","r") as csvFile:
    csvReader = csv.reader(csvFile)
    header = next(csvReader)
    for row in csvReader:
        # 'row' is a list of elements (fields) for each row
        maybeValid = StockPrice.parseCSV(row)
        if maybeValid is None:
            print(f"skipping row: {row}")
        else: allStocks.append(maybeValid)
priceByName: defaultdict[str, List[StockPrice]] = defaultdict(list)

for stock in allStocks:
    priceByName[stock.name].append(stock)
    
print(priceByName)