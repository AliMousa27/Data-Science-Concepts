users = [[0, "Hero", 0],
         [1, "Dunn", 2],
         [2, "Sue", 3],
         [3, "Chi", 3]]

from typing import Tuple, Sequence, List, Any, Callable, Dict, Iterator
from collections import defaultdict

# A few type aliases we'll use later
Row = Dict[str, Any]                        # A database row
WhereClause = Callable[[Row], bool]         # Predicate for a single row
HavingClause = Callable[[List[Row]], bool]  # Predicate over multiple rows

class Table:
    def __init__(self, columns: List[str], types: List[type]) -> None:
        assert len(columns) == len(types), "# of columns must == # of types"

        self.columns = columns         # Names of columns
        self.types = types             # Data types of columns
        self.rows: List[Row] = []      # (no data yet)

    def col2type(self, col: str) -> type:
        idx = self.columns.index(col)      # Find the index of the column,
        return self.types[idx]             # and return its type.

    def insert(self, values: List[Any]) -> None:
        # Check for right # of values
        if len(values) != len(self.types):
            raise ValueError(f"You need to provide {len(self.types)} values")

        # Check for right types of values
        for value, typ3 in zip(values, self.types):
            if not isinstance(value, typ3) and value is not None:
                raise TypeError(f"Expected type {typ3} but got {value}")

        # Add the corresponding dict as a "row"
        self.rows.append(dict(zip(self.columns, values)))

    def __getitem__(self, idx: int) -> Row:
        return self.rows[idx]

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self):
        """Pretty representation of the table: columns then rows"""
        rows = "\n".join(str(row) for row in self.rows)

        return f"{self.columns}\n{rows}"

    def update(self,
               updates: Dict[str, Any],
               predicate: WhereClause = lambda row: True):
        # First make sure the updates have valid names and types
        for column, newValue in updates.items():
            if column not in self.columns:
                raise ValueError(f"invalid column: {column}")

            typ3 = self.col2type(column)
            if not isinstance(newValue, typ3) and newValue is not None:
                raise TypeError(f"expected type {typ3}, but got {newValue}")

        # Now update
        for row in self.rows:
            if predicate(row):
                for column, newValue in updates.items():
                    row[column] = newValue

    def delete(self, predicate: WhereClause = lambda row: True) -> None:
        """Delete all rows matching predicate"""
        self.rows = [row for row in self.rows if not predicate(row)]

    def select(self,
               keepColumns: List[str] = None,
               additionalColumns: Dict[str, Callable] = None) -> 'Table':

        if keepColumns is None:         # If no columns specified,
            keepColumns = self.columns  # return all columns

        if additionalColumns is None:
            additionalColumns = {}

        # New column names and types
        newColumns = keepColumns + list(additionalColumns.keys())
        keepTypes = [self.col2type(col) for col in keepColumns]

        # This is how to get the return type from a type annotation.
        # It will crash if `calculation` doesn't have a return type.
        addTypes = [calculation.__annotations__['return']
                     for calculation in additionalColumns.values()]

        # Create a new table for results
        newTable = Table(newColumns, keepTypes + addTypes)

        for row in self.rows:
            newRow = [row[column] for column in keepColumns]
            for columnName, calculation in additionalColumns.items():
                newRow.append(calculation(row))
            newTable.insert(newRow)

        return newTable

    def where(self, predicate: WhereClause = lambda row: True) -> 'Table':
        """Return only the rows that satisfy the supplied predicate"""
        whereTable = Table(self.columns, self.types)
        for row in self.rows:
            if predicate(row):
                values = [row[column] for column in self.columns]
                whereTable.insert(values)
        return whereTable

    def limit(self, numRows: int) -> 'Table':
        """Return only the first `numRows` rows"""
        limitTable = Table(self.columns, self.types)
        for i, row in enumerate(self.rows):
            if i >= numRows:
                break
            values = [row[column] for column in self.columns]
            limitTable.insert(values)
        return limitTable

    def groupBy(self,
                 groupByColumns: List[str],
                 aggregates: Dict[str, Callable],
                 having: HavingClause = lambda group: True) -> 'Table':

        groupedRows = defaultdict(list)

        # Populate groups
        for row in self.rows:
            key = tuple(row[column] for column in groupByColumns)
            groupedRows[key].append(row)

        # Result table consists of groupBy columns and aggregates
        newColumns = groupByColumns + list(aggregates.keys())
        groupByTypes = [self.col2type(col) for col in groupByColumns]
        aggregateTypes = [agg.__annotations__['return']
                           for agg in aggregates.values()]
        resultTable = Table(newColumns, groupByTypes + aggregateTypes)

        for key, rows in groupedRows.items():
            if having(rows):
                newRow = list(key)
                for aggregateName, aggregateFn in aggregates.items():
                    newRow.append(aggregateFn(rows))
                resultTable.insert(newRow)

        return resultTable

    def orderBy(self, order: Callable[[Row], Any]) -> 'Table':
        newTable = self.select()       # make a copy
        newTable.rows.sort(key=order)
        return newTable

    def join(self, otherTable: 'Table', leftJoin: bool = False) -> 'Table':

        joinOnColumns = [c for c in self.columns           # columns in
                         if c in otherTable.columns]      # both tables

        additionalColumns = [c for c in otherTable.columns # columns only
                              if c not in joinOnColumns]   # in right table

        # all columns from left table + additionalColumns from right table
        newColumns = self.columns + additionalColumns
        newTypes = self.types + [otherTable.col2type(col)
                                for col in additionalColumns]

        joinTable = Table(newColumns, newTypes)

        for row in self.rows:
            def isJoin(otherRow):
                return all(otherRow[c] == row[c] for c in joinOnColumns)

            otherRows = otherTable.where(isJoin).rows

            # Each other row that matches this one produces a result row.
            for otherRow in otherRows:
                joinTable.insert([row[c] for c in self.columns] +
                                 [otherRow[c] for c in additionalColumns])

            # If no rows match and it's a left join, output with Nones.
            if leftJoin and not otherRows:
                joinTable.insert([row[c] for c in self.columns] +
                                 [None for c in additionalColumns])

        return joinTable


def main():
    # Constructor requires column names and types
    users = Table(['user_id', 'name', 'num_friends'], [int, str, int])
    users.insert([0, "Hero", 0])
    users.insert([1, "Dunn", 2])
    users.insert([2, "Sue", 3])
    users.insert([3, "Chi", 3])
    users.insert([4, "Thor", 3])
    users.insert([5, "Clive", 2])
    users.insert([6, "Hicks", 3])
    users.insert([7, "Devin", 2])
    users.insert([8, "Kate", 2])
    users.insert([9, "Klein", 3])
    users.insert([10, "Jen", 1])

    assert len(users) == 11
    assert users[1]['name'] == 'Dunn'

    assert users[1]['num_friends'] == 2             # Original value

    users.update({'num_friends' : 3},               # Set num_friends = 3
                 lambda row: row['user_id'] == 1)   # in rows where user_id == 1

    assert users[1]['num_friends'] == 3             # Updated value

    # SELECT * FROM users;
    allUsers = users.select()
    assert len(allUsers) == 11

    # SELECT * FROM users LIMIT 2;
    twoUsers = users.limit(2)
    assert len(twoUsers) == 2

    # SELECT user_id FROM users;
    justIds = users.select(keepColumns=["user_id"])
    assert justIds.columns == ['user_id']

    # SELECT user_id FROM users WHERE name = 'Dunn';
    dunnIds = (
        users
        .where(lambda row: row["name"] == "Dunn")
        .select(keepColumns=["user_id"])
    )
    assert len(dunnIds) == 1
    assert dunnIds[0] == {"user_id": 1}

    # SELECT LENGTH(name) AS name_length FROM users;
    def nameLength(row) -> int: return len(row["name"])

    nameLengths = users.select(keepColumns=[],
                               additionalColumns={"name_length": nameLength})
    assert nameLengths[0]['name_length'] == len("Hero")

    def minUserId(rows) -> int:
        return min(row["user_id"] for row in rows)

    def length(rows) -> int:
        return len(rows)

    statsByLength = (
        users
        .select(additionalColumns={"name_length" : nameLength})
        .groupBy(groupByColumns=["name_length"],
                  aggregates={"min_user_id" : minUserId,
                              "num_users" : length})
    )

    assert len(statsByLength) == 3
    assert statsByLength.columns == ["name_length", "min_user_id", "num_users"]

    def firstLetterOfName(row: Row) -> str:
        return row["name"][0] if row["name"] else ""

    def averageNumFriends(rows: List[Row]) -> float:
        return sum(row["num_friends"] for row in rows) / len(rows)

    def enoughFriends(rows: List[Row]) -> bool:
        return averageNumFriends(rows) > 1

    avgFriendsByLetter = (
        users
        .select(additionalColumns={'first_letter' : firstLetterOfName})
        .groupBy(groupByColumns=['first_letter'],
                  aggregates={"avg_num_friends" : averageNumFriends},
                  having=enoughFriends)
    )

    assert len(avgFriendsByLetter) == 6
    assert {row['first_letter'] for row in avgFriendsByLetter} == \
           {"H", "D", "S", "C", "T", "K"}

    def sumUserIds(rows: List[Row]) -> int:
        return sum(row["user_id"] for row in rows)

    userIdSum = (
        users
        .where(lambda row: row["user_id"] > 1)
        .groupBy(groupByColumns=[],
                  aggregates={ "user_id_sum" : sumUserIds })
    )

    assert len(userIdSum) == 1
    assert userIdSum[0]["user_id_sum"] == 54

    friendliestLetters = (
        avgFriendsByLetter
        .orderBy(lambda row: -row["avg_num_friends"])
        .limit(4)
    )

    assert len(friendliestLetters) == 4
    assert friendliestLetters[0]['first_letter'] in ['S', 'T']

    userInterests = Table(['user_id', 'interest'], [int, str])
    userInterests.insert([0, "SQL"])
    userInterests.insert([0, "NoSQL"])
    userInterests.insert([2, "SQL"])
    userInterests.insert([2, "MySQL"])

    sqlUsers = (
        users
        .join(userInterests)
        .where(lambda row: row["interest"] == "SQL")
        .select(keepColumns=["name"])
    )

    assert len(sqlUsers) == 2
    sqlUserNames = {row["name"] for row in sqlUsers}
    assert sqlUserNames == {"Hero", "Sue"}

    def countInterests(rows: List[Row]) -> int:
        """counts how many rows have non-None interests"""
        return len([row for row in rows if row["interest"] is not None])

    userInterestCounts = (
    users
    .join(userInterests, leftJoin=True)
    .groupBy(groupByColumns=["user_id"],
              aggregates={"num_interests" : countInterests})
)

    likesSqlUserIds = (
        userInterests
        .where(lambda row: row["interest"] == "SQL")
        .select(keepColumns=['user_id'])
    )

    likesSqlUserIds.groupBy(groupByColumns=[],
                            aggregates={"min_user_id" : minUserId})

    assert len(likesSqlUserIds) == 2

    (
        userInterests
        .where(lambda row: row["interest"] == "SQL")
        .join(users)
        .select(["name"])
    )

    (
        userInterests
        .join(users)
        .where(lambda row: row["interest"] == "SQL")
        .select(["name"])
    )

if __name__ == "__main__":
    main()

    
if __name__ == "__main__": main()