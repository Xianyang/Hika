import pandas as pd
import numpy as np
import os
# from talib import STOCH
from pytz import timezone
from datetime import datetime, date, time, timedelta
# from dateutil.relativedelta import relativedelta
import threading
import time as timeToCount

# ----------------------Constants-----------------------
# timezone
GMT = timezone('GMT+0')
timezoneList = {'HKE': timezone('Asia/Hong_Kong'), 'NYMEX': timezone('US/Eastern'), 'TSX': timezone('Canada/Eastern'),
                'ASX': timezone('Australia/Sydney'), 'FWB': timezone('CET'), 'LSE': timezone('Europe/London'),
                'NYSE': timezone('US/Eastern')}
closingTime = {'HKE': time(16), 'NYMEX': time(17, 15), 'TSX': time(16), 'ASX': time(16), 'FWB': time(20),
               'LSE': time(16, 30), 'NYSE': time(16)}

# matFile = "./Nicole/CL1 COMDTY Matsuba/CL1 COMDTY_res2.csv"
matFile = './Data/CL1 COMDTY_res2.csv'

HOST = "localhost"
PORT = 8194

amount = {0:1, 1:1, 2:2, 3:4, 4:8}
unit = 10

# -------------------end of constants--------------------


class Strategy(threading.Thread):
    def __init__(self, poslimit, capital, stoploss, startdate, enddate, mat_dateList, mat_value):
        threading.Thread.__init__(self)
        self.poslimit = poslimit
        self.capital = capital
        self.stoploss = stoploss
        for i in range(1, len(mat_dateList)):
            dt_last = mat_dateList[i - 1]
            dt = mat_dateList[i]
            if dt_last < startdate <= dt:
                self.startdate = dt
                sindex = i
            if dt_last <= enddate < dt:
                self.enddate = dt_last
                eindex = i
                break
        self.matdate = mat_dateList[sindex:eindex]
        self.matvalue = mat_value
        # self.marketFile = "./Nicole/MarketData/CL1_5min.csv"
        self.marketFile = "./Data/CL1 COMDTY_2016-01-01_2016-06-01_5Minutes.csv"
        self.resultPath = "./Output/" + startdate.strftime('%Y%m%d') + '_' + enddate.strftime('%Y%m%d') + '_Unit' + str(unit) + '/'

    def prepareDirectory(self):  # prepare both database and backup folders
        if not os.path.exists(self.resultPath):
            try:
                os.makedirs(self.resultPath)
            except OSError:
                print self.resultPath + " directory could not be created!"
                return False
        return True

    def readMarket(self):
        csvfile = pd.read_csv(self.marketFile)

        # these dics save high and low data--{dt:high data}
        highDict, lowDict = {}, {}

        # this list saves all the datetime value from start date to end date
        datetimeList = []
        startDatetime = datetime.combine(self.startdate, time(18))
        endDatetime = datetime.combine(self.enddate, time(17, 15))
        for i in csvfile.index:
            dt = datetime.strptime(csvfile.loc[i, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
            if startDatetime <= dt <= endDatetime:
                datetimeList.append(dt)
                highDict[dt] = csvfile.loc[i, 'HIGH']
                lowDict[dt] = csvfile.loc[i, 'LOW']
            if dt > datetime.combine(self.enddate, time(17, 15)):
                break

        print ('%s seconds to read market file' % (timeToCount.time() - programStartTime))
        self.netPos = 0
        self.shortPos = 0
        shortInfo, shortTrade, shortPNL = [], [], []
        shortCF = 0.0
        self.longPos = 0
        longInfo, longTrade, longPNL = [], [], []
        longCF = 0.0
        totalResult = []
        [mat_high, mat_low] = self.matvalue[self.startdate]
        target_high, target_low = [], []

        # this is a list save five levels from the high matsuba and low matsuba
        for i in range(5):
            # target_high.append([mat_high + i, 0 - (i * 10 + 10)])
            # target_low.append([mat_low - i, i * 10 + 10])
            target_high.append([mat_high + i, -amount[i] * unit])
            target_low.append([mat_low - i, amount[i] * unit])

        shortpnl, longpnl, shortreturn, longreturn = 0.0, 0.0, 0.0, 0.0

        short_exe, long_exe = [], []
        for dt in datetimeList:
            # calculate the exit price for short and long
            if self.shortPos != 0:
                short_exitprice = abs(shortCF / self.shortPos) * (1 - self.stoploss)
            else:
                short_exitprice = 0
            if self.longPos != 0:
                long_exitprice = abs(longCF / self.longPos) * (1 + self.stoploss)
            else:
                long_exitprice = 9999

            # it means that mat_high and mat_low does not exist at this day
            if mat_high == 9999:
                if dt.time() >= time(18):
                    opent = dt.date()
                else:
                    opent = (dt - timedelta(1)).date()
                if opent in self.matdate:
                    mat_high = self.matvalue[opent][0]
                    # target_high = [[mat_high + i, 0 - (i * 10 + 10)] for i in range(5)]
                    target_high = [[mat_high + i, -amount[i] * unit] for i in range(5)]

            if mat_low == 0:
                if dt.time() >= time(18):
                    opent = dt.date()
                else:
                    opent = (dt - timedelta(1)).date()
                if opent in self.matdate:
                    mat_low = self.matvalue[opent][1]
                    # target_low = [[mat_low - i, (i * 10 + 10)] for i in range(5)]
                    target_low = [[mat_low - i, amount[i] * unit] for i in range(5)]

            # short bias, using target_high
            highPriceForDt = highDict[dt]
            lowPriceForDt = lowDict[dt]
            # the info dic is [dt, high_value, low_value, target1, target2, target3, target4, target5, size, position, exit_price]
            shortInfo.append(
                [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_high] + [0, self.shortPos, short_exitprice])
            # if the candle covers the target, then add the target to short exercise
            for price, size in target_high:
                if lowPriceForDt <= price <= highPriceForDt:
                    short_exe.append([price, size])  # price > 0, size < 0
            if len(short_exe) != 0:
                for price, size in short_exe:
                    if abs(self.netPos + size) <= self.poslimit:
                        self.shortPos += size
                        self.netPos += size
                        shortCF += 0 - price * size  # shortCF > 0
                        shortInfo[-1][8] += size
                        shortTrade.append([dt, price, size])
                        while target_high.index([price, size]) != 0:
                            del target_high[0]
                            # target_high.append([target_high[-1][0] + 1, target_high[-1][1] - 10])
                            target_high.append([target_high[-1][0] + 1, target_high[-1][1] * 2])
                        target_high.remove([price, size])
                        target_high.append([target_high[-1][0] + 1, target_high[-1][1] * 2])
                    else:
                        newsize = self.poslimit - self.netPos
                        if newsize > 0.0:
                            self.shortPos += newsize
                            self.netPos += newsize
                            shortCF += 0 - price * newsize  # shortCF > 0
                            shortInfo[-1][8] += newsize
                            shortTrade.append([dt, price, newsize])
                            target_high[target_high.index([price, size])][1] = size - newsize
                        break
                short_exitprice = abs(shortCF / self.shortPos) * (1 - self.stoploss)
                short_exe = []
                shortInfo[-1][9] = self.shortPos
                shortInfo[-1][10] = short_exitprice

            # long bias, using target_low
            longInfo.append(
                [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_low] + [0, self.longPos, long_exitprice])
            for price, size in target_low:
                if lowPriceForDt <= price <= highPriceForDt:
                    long_exe.append([price, size])  # price > 0, size > 0
            if len(long_exe) != 0:
                for price, size in long_exe:
                    if abs(self.netPos + size) <= self.poslimit:
                        self.longPos += size
                        self.netPos += size
                        longCF += 0 - price * size
                        longInfo[-1][8] += size
                        longTrade.append([dt, price, size])
                        while target_low.index([price, size]) != 0:
                            del target_low[0]
                            # target_low.append([target_low[-1][0] - 1, target_low[-1][1] + 10])
                            target_low.append([target_low[-1][0] - 1, target_low[-1][1] * 2])
                        target_low.remove([price, size])
                        # target_low.append([target_low[-1][0] - 1, target_low[-1][1] + 10])
                        target_low.append([target_low[-1][0] - 1, target_low[-1][1] * 2])
                    else:
                        newsize = self.poslimit - self.netPos
                        if newsize > 0.0:
                            self.longPos += newsize
                            self.netPos += newsize
                            longCF += 0 - price * newsize
                            longInfo[-1][8] += newsize
                            longTrade.append([dt, price, newsize])
                            target_low[target_low.index([price, size])][1] = size - newsize
                        break
                long_exitprice = abs(longCF / self.longPos) * (1 + self.stoploss)
                long_exe = []
                longInfo[-1][9] = self.longPos
                longInfo[-1][10] = long_exitprice

            # short exit
            if lowPriceForDt <= short_exitprice:
                exitorder = 0 - self.shortPos
                self.shortPos = 0
                self.netPos += exitorder
                shortpnl += shortCF - exitorder * short_exitprice
                shortreturn += 1000 * shortpnl / self.capital
                shortPNL.append([dt, shortCF, exitorder, short_exitprice, shortreturn])
                shortInfo.append(
                    [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_high] + [exitorder, self.shortPos, short_exitprice])
                shortTrade.append([dt, short_exitprice, exitorder])
                shortCF = 0.0
                if dt.time() >= time(18):
                    opent = dt.date()
                else:
                    opent = (dt - timedelta(1)).date()
                if opent in self.matdate:
                    mat_high = self.matvalue[opent][0]
                    # target_high = [[mat_high + i, 0 - (i * 10 + 10)] for i in range(5)]
                    target_high = [[mat_high + i, -amount[i] * unit] for i in range(5)]
                    for item in target_high:
                        if item[0] <= highPriceForDt:
                            target_high.remove(item)
                            # target_high.append([target_high[-1][0] + 1, target_high[-1][1] - 10])
                            target_high.append([target_high[-1][0] + 1, target_high[-1][1] * 2])
                        else:
                            break
                else:
                    mat_high = 9999
                    target_high = [[9999, 0] for i in range(5)]

            # long exit
            if highPriceForDt >= long_exitprice:
                exitorder = 0 - self.longPos
                self.longPos = 0
                self.netPos += exitorder
                longpnl += longCF - exitorder * long_exitprice
                longreturn += 1000 * longpnl / self.capital
                longPNL.append([dt, longCF, exitorder, long_exitprice, longreturn])
                longInfo.append([dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_low] + [exitorder, self.longPos,
                                                                                                long_exitprice])
                longTrade.append([dt, long_exitprice, exitorder])
                longCF = 0.0
                if dt.time() >= time(18):
                    opent = dt.date()
                else:
                    opent = (dt - timedelta(1)).date()
                if opent in self.matdate:
                    mat_low = self.matvalue[opent][1]
                    # target_low = [[mat_low - i, i * 10 + 10] for i in range(5)]
                    target_low = [[mat_low - i, amount[i] * unit] for i in range(5)]
                    for item in target_low:
                        if item[0] >= lowPriceForDt:
                            target_low.remove(item)
                            # target_low.append([target_low[-1][0] - 1, target_low[-1][1] + 10])
                            target_low.append([target_low[-1][0] - 1, target_low[-1][1] * 2])
                        else:
                            break
                else:
                    mat_low = 0
                    target_low = [[0, 0] for i in range(5)]

            # the last day
            if datetimeList.index(dt) == len(datetimeList) - 1:
                shortPNL.append([dt, shortCF, self.shortPos, 0, 0])
                longPNL.append([dt, longCF, self.longPos, 0, 0])
            else:
                if dt.time() < time(18) <= datetimeList[datetimeList.index(dt) + 1].time():
                    totalResult.append(
                        [dt, self.shortPos, self.longPos, self.netPos, shortpnl, longpnl, shortreturn, longreturn])
                    shortpnl, longpnl, shortreturn, longreturn = 0, 0, 0, 0

        count = 0
        dt = self.startdate
        while dt < self.enddate:
            opent = datetime.combine(dt, time(18))
            closet = datetime.combine(dt, time(17, 15)) + timedelta(1)
            for item in datetimeList:
                if opent <= item < closet:
                    count += 1
                    break
            dt += timedelta(1)

        # 1
        shortInfofile = pd.DataFrame(shortInfo, columns=['Date', 'high price', 'low price', 'target high level1',
                                                         'target high level2', 'target high level3',
                                                         'target high level4', 'target high level5', 'order size',
                                                         'accumulated short position', 'short exit price'])
        shortInfofile.to_csv(self.resultPath + "5min_tradeinfo_short.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 2
        shortTradefile = pd.DataFrame(shortTrade, columns=['date', 'price', 'order'])
        shortTradefile.to_csv(self.resultPath + "5min_tradelog_short.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 3
        shortpnlfile = pd.DataFrame(shortPNL, columns=['date', 'short cashflow', 'exit order', 'exit price', 'return'])
        shortpnlfile.to_csv(self.resultPath + "5min_pnl_short.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 4
        longInfofile = pd.DataFrame(longInfo, columns=['Date', 'high price', 'low price', 'target low level1',
                                                       'target low level2', 'target low level3', 'target low level4',
                                                       'target low level5', 'order size', 'accumulated long position',
                                                       'long exit price'])
        longInfofile.to_csv(self.resultPath + "5min_tradeinfo_long.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 5
        longTradefile = pd.DataFrame(longTrade, columns=['date', 'price', 'order'])
        longTradefile.to_csv(self.resultPath + "5min_tradelog_long.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 6
        longpnlfile = pd.DataFrame(longPNL, columns=['date', 'long cashflow', 'exit order', 'exit price', 'return'])
        longpnlfile.to_csv(self.resultPath + "5min_pnl_long.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 7
        totalpnl = pd.DataFrame(totalResult,
                                columns=['date', 'short position', 'long position', 'net position', 'short pnl',
                                         'long pnl', 'short return', 'long return'])
        totalpnl.to_csv(self.resultPath + "5min_totalreturn.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

    def run(self):
        if not self.prepareDirectory():
            return
        self.readMarket()


def readMatsuba(filename):
    mat_dateList, mat_value = [], {}
    csvfile = pd.read_csv(filename)
    for i in csvfile.index:
        dt = datetime.strptime(csvfile.loc[i, 'DATE'], '%Y-%m-%d').date()
        mat_dateList.append(dt)
        mat_value[dt] = [csvfile.loc[i, 'HIGH'], csvfile.loc[i, 'LOW']]
    return mat_dateList, mat_value


if __name__ == "__main__":
    # convertTime(marketFile)   # convert time in excel bloomberg to local time and generate new market data file
    # parameters
    poslimit = 600
    capital = 5000.0 * poslimit
    stoploss = 0.03
    startdate = datetime(2016, 1, 1).date()
    enddate = datetime(2016, 5, 30).date()
    mat_dateList, mat_value = readMatsuba(matFile)

    programStartTime = timeToCount.time()

    print "dMatonly_5min.py running..."
    print "start date: " + startdate.strftime('%Y_%m_%d') + " end data: " + enddate.strftime('%Y_%m_%d')

    threads = []
    '''
    for dt in mat_dateList:
        if dt>=testing_start and dt<=testing_end:
            task = Strategy(dt, mat_value[dt], poslimit)
            threads.append(task)
        if dt>testing_end:
            break
    '''
    task = Strategy(poslimit, capital, stoploss, startdate, enddate, mat_dateList, mat_value)
    threads.append(task)

    if len(threads) > 0:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        print "no date starts."
    print "Waiting for finish..."
    print ('This program take %s seconds to run' % (timeToCount.time() - programStartTime))

