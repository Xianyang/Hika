import pickle
import time as ti
from datetime import datetime, time, timedelta
import csv

with open('test.csv', 'wb') as f:
    a = csv.writer(f)
    a.writerow('a')

with open('test.csv', 'wb') as f:
    a = csv.writer(f)
    a.writerow('b')

def getOpenDateForADatetime(dt):
    if time(18) <= dt.time():
        return dt.date()
    else:
        return (dt - timedelta(1)).date()

nowDatetime = datetime.now()
openDate = getOpenDateForADatetime(nowDatetime)


try:
    with open('store.pckl') as f:
        a, b, c = pickle.load(f)
        print a, b, c
except IOError:
    print 'no such file'



with open('store.pckl', 'w') as f:
    pickle.dump([1, 2, 3], f)