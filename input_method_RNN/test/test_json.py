import pandas as pd

dfjo = pd.DataFrame(
    dict(A = range(1,4),B = range(4,7),C = range(7,10)),
    columns = list('ABC'),
    index = list('XYZ'),
)
print(dfjo)

dfjo.to_json('df.json',orient='records',lines = True)
# orient='columns'---{"A":{"X":1,"Y":2,"Z":3},"B":{"X":4,"Y":5,"Z":6},"C":{"X":7,"Y":8,"Z":9}}
# orient='index'-----{"X":{"A":1,"B":4,"C":7},"Y":{"A":2,"B":5,"C":8},"Z":{"A":3,"B":6,"C":9}}
# orient='split'-----{"columns":["A","B","C"],"index":["X","Y","Z"],"data":[[1,4,7],[2,5,8],[3,6,9]]}
# orient='records'---[{"A":1,"B":4,"C":7},{"A":2,"B":5,"C":8},{"A":3,"B":6,"C":9}]   数组，按行来
# orient='values'----[[1,4,7],[2,5,8],[3,6,9]]   只有值，按行
# 只有在 orient='records'时，设置lines = True          {"A":1,"B":4,"C":7}
#                                                   {"A":2,"B":5,"C":8}
#                                                   {"A":3,"B":6,"C":9}

# jsonl 文件每行是一个json对象, 所以需要设置lines = True,按行读取
