import csv
import pandas
import numpy

a = pandas.read_csv('./ParaAndOutputForPaperTrading/5min_pnl_long.csv').values.tolist()
a.append(a[0])

shortpnlfile = pandas.DataFrame(a, columns=['date', 'short cashflow', 'exit order', 'exit price', 'return'])
shortpnlfile.to_csv("./ParaAndOutputForPaperTrading/b.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

