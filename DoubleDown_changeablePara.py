import pandas as pd
from scipy.optimize import minimize
import numpy as np
import os
from datetime import datetime, time, timedelta
import threading
import time as timeToCount


class Strategy(threading.Thread):
    def __init__(self, poslimit, capital, takeProfit, startdate, enddate, mat_dateList, mat_value):
        # threading.Thread.__init__(self)
        self.poslimit = poslimit
        self.capital = capital
        self.takeProfit = takeProfit
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
        self.marketFile = "./Data/CL1 COMDTY_2016-01-01_2016-06-01_5Minutes.csv"
        self.resultPath = './OutputCP/Unit' + str(unit) + '_' + startdate.strftime('%Y%m%d') + '_' + enddate.strftime('%Y%m%d') + '/'

    def prepareDirectory(self):  # prepare both database and backup folders
        if not os.path.exists(self.resultPath):
            try:
                os.makedirs(self.resultPath)
            except OSError:
                print self.resultPath + " directory could not be created!"
                return False
        return True

    def resetTargetList(self, matValue, indicator):
        targetList = []
        if indicator == 'High':
            for i in xrange(roundForLongAndShort):
                if i == 0:
                    targetList.append([matValue, -amount[i] * unit])
                else:
                    targetList.append([targetList[-1][0] * (1 + percentForALevel), -amount[i] * unit])
        elif indicator == 'Low':
            for i in xrange(roundForLongAndShort):
                if i == 0:
                    targetList.append([matValue, amount[i] * unit])
                else:
                    targetList.append([targetList[-1][0] * (1 - percentForALevel), amount[i] * unit])
        else:
            raise ValueError('indicator is invalid')
        return targetList

    def readMarket(self):
        '''
        if len(amount) < roundForLongAndShort or roundForLongAndShort < 2:
            print 'invalid round %d' % roundForLongAndShort
            return -1
        print 'the stop loss round is %d' % roundForLongAndShort
        '''

        startDatetime = datetime.combine(self.startdate, time(18))
        endDatetime = datetime.combine(self.enddate, time(17, 15))

        # parameters
        self.netPos, self.shortPos, self.longPos = 0, 0, 0
        shortCF, longCF = 0.0, 0.0
        shortInfo, shortTrade, shortPNL = [], [], []
        longInfo, longTrade, longPNL = [], [], []
        totalResult = []
        [mat_high, mat_low] = self.matvalue[self.startdate]
        short_exe, long_exe = [], []
        shortpnl, longpnl, shortreturn, longreturn = None, None, None, None
        accumulateReturn = 0

        # this is a list saving five levels from the high matsuba and low matsuba
        target_high = self.resetTargetList(mat_high, 'High')
        target_low = self.resetTargetList(mat_low, 'Low')

        csvfile = pd.read_csv(self.marketFile)
        for index in csvfile.index:
            dt = datetime.strptime(csvfile.loc[index, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
            if dt < startDatetime:
                continue
            elif endDatetime < dt:
                break
            elif startDatetime <= dt <= endDatetime:
                highPriceForDt = csvfile.loc[index, 'HIGH']
                lowPriceForDt = csvfile.loc[index, 'LOW']
                shortStopLoss, longStopLoss = False, False

                # calculate the take profit price for short and long
                if self.shortPos != 0:
                    short_takePorfit_price = abs(shortCF / self.shortPos) * (1 - self.takeProfit)
                else:
                    short_takePorfit_price = None
                if self.longPos != 0:
                    long_takePorfit_price = abs(longCF / self.longPos) * (1 + self.takeProfit)
                else:
                    long_takePorfit_price = None

                # it means that mat_high and mat_low does not exist at this day
                if mat_high is None:
                    if dt.time() >= time(18):
                        opent = dt.date()
                    else:
                        opent = (dt - timedelta(1)).date()
                    if opent in self.matdate:
                        mat_high = self.matvalue[opent][0]
                        target_high = self.resetTargetList(mat_high, 'High')

                if mat_low is None:
                    if dt.time() >= time(18):
                        opent = dt.date()
                    else:
                        opent = (dt - timedelta(1)).date()
                    if opent in self.matdate:
                        mat_low = self.matvalue[opent][1]
                        target_low = self.resetTargetList(mat_low, 'Low')

                # short bias, using target_high
                # the info dic is [dt, high_value, low_value, target1, target2, target3, target4, target5, size, position, exit_price]
                shortInfo.append(
                    [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_high] + [0, self.shortPos, short_takePorfit_price])
                # if the candle covers the target, then add the target to short exercise or exit
                for index, [price, size] in enumerate(target_high):
                    if price and lowPriceForDt <= price <= highPriceForDt:
                        # if the price hits the last level, then stop loss!
                        if index == len(target_high) - 1:
                            shortStopLoss = True
                            break
                        else:
                            short_exe.append([price, size])  # price > 0, size < 0
                            # after exercise, set the value of the level to None
                            target_high[index] = [None, 0]

                # short exercise
                if len(short_exe) != 0 and shortStopLoss is False:
                    for price, size in short_exe:
                        # hit the limit position, set a new size
                        if abs(self.netPos + size) > self.poslimit:
                            size = self.poslimit - self.netPos
                            if size == 0:
                                continue
                            print 'short exercise hits the position limit and the new size is %d' % size
                        self.shortPos += size
                        self.netPos += size
                        shortCF += 0 - price * size  # shortCF > 0
                        shortInfo[-1][-3] += size
                        shortTrade.append([dt, price, size])
                        print 'short exercise at $%.2f for %d position on ' % (price, size) + dt.strftime('%Y-%m-%d %H:%M')

                    short_takePorfit_price = abs(shortCF / self.shortPos) * (1 - self.takeProfit)
                    short_exe = []
                    shortInfo[-1][-2] = self.shortPos
                    shortInfo[-1][-1] = short_takePorfit_price

                # long bias, using target_low
                # the info dic is [dt, high_value, low_value, target1, target2, target3, target4, target5, size, position, exit_price]
                longInfo.append(
                    [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_low] + [0, self.longPos, long_takePorfit_price])
                # if the candle covers the target, then add the target to short exercise or exit
                for index, [price, size] in enumerate(target_low):
                    if price and lowPriceForDt <= price <= highPriceForDt:
                        # if the price hits the last level, then stop loss!
                        if index == len(target_low) - 1:
                            longStopLoss = True
                            break
                        else:
                            long_exe.append([price, size])  # price > 0, size > 0
                            # after exercise, set the value of the level to None
                            target_low[index] = [None, 0]

                # long exercise
                if len(long_exe) != 0 and longStopLoss is False:
                    for price, size in long_exe:
                        # hit the limit position, set a new size
                        if abs(self.netPos + size) > self.poslimit:
                            size = self.poslimit - self.netPos
                            if size == 0:
                                continue
                            print 'long exercise hits the position limit and the new size is %d' % size
                        self.longPos += size
                        self.netPos += size
                        longCF += 0 - price * size  # longCF > 0
                        longInfo[-1][-3] += size
                        longTrade.append([dt, price, size])
                        print 'long exercise at $%.2f for %d position on ' % (price, size) + dt.strftime('%Y-%m-%d %H:%M')

                    long_takePorfit_price = abs(longCF / self.longPos) * (1 + self.takeProfit)
                    long_exe = []
                    longInfo[-1][-2] = self.longPos
                    longInfo[-1][-1] = long_takePorfit_price

                # short exit
                if (short_takePorfit_price and lowPriceForDt <= short_takePorfit_price) or shortStopLoss is True:
                    if lowPriceForDt <= short_takePorfit_price:
                        exitPrice = short_takePorfit_price
                    elif shortStopLoss is True:
                        exitPrice = lowPriceForDt

                    exitorder = 0 - self.shortPos
                    self.shortPos = 0
                    self.netPos += exitorder
                    shortpnl = shortCF - exitorder * exitPrice
                    shortreturn = 1000 * shortpnl / self.capital
                    accumulateReturn += shortreturn
                    shortPNL.append([dt, shortCF, exitorder, exitPrice, shortreturn])
                    shortInfo.append(
                        [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_high] + [exitorder, self.shortPos, exitPrice])
                    shortTrade.append([dt, exitPrice, exitorder])
                    shortCF = 0.0
                    print 'short exit at $%.2f, return is %.2f' % (exitPrice, shortreturn * 100) + '% on ' + dt.strftime('%Y-%m-%d %H:%M')
                    if dt.time() >= time(18):
                        opent = dt.date()
                    else:
                        opent = (dt - timedelta(1)).date()
                    if opent in self.matdate:
                        mat_high = self.matvalue[opent][0]
                        target_high = self.resetTargetList(mat_high, 'High')
                        while target_high[0][0] <= highPriceForDt:
                            target_high = self.resetTargetList(target_high[0][0] * (1 + percentForALevel), 'High')
                    else:
                        mat_high = None
                        target_high = [[None, 0] for i in range(roundForLongAndShort)]

                # long exit
                if (long_takePorfit_price and long_takePorfit_price <= highPriceForDt) or longStopLoss is True:
                    if long_takePorfit_price <= highPriceForDt:
                        exitPrice = long_takePorfit_price
                    elif longStopLoss is True:
                        exitPrice = highPriceForDt

                    exitorder = 0 - self.longPos
                    self.longPos = 0
                    self.netPos += exitorder
                    longpnl = longCF - exitorder * exitPrice
                    longreturn = 1000 * longpnl / self.capital
                    accumulateReturn += longreturn
                    longPNL.append([dt, longCF, exitorder, exitPrice, longreturn])
                    longInfo.append(
                        [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_low] + [exitorder, self.longPos, exitPrice])
                    longTrade.append([dt, exitPrice, exitorder])
                    longCF = 0.0
                    print 'long exit at $%.2f, return is %.2f' % (exitPrice, longreturn * 100) + '% on ' + dt.strftime('%Y-%m-%d %H:%M')
                    if dt.time() >= time(18):
                        opent = dt.date()
                    else:
                        opent = (dt - timedelta(1)).date()
                    if opent in self.matdate:
                        mat_low = self.matvalue[opent][1]
                        target_low = self.resetTargetList(mat_low, 'Low')
                        while lowPriceForDt <= target_low[0][0]:
                            target_low = self.resetTargetList(target_low[0][0] * (1 - percentForALevel), 'Low')
                    else:
                        mat_low = None
                        target_low = [[None, 0] for i in range(roundForLongAndShort)]

                # the last day
                if index == csvfile.index.values[-1]:
                    shortPNL.append([dt, shortCF, self.shortPos, 0, 0])
                    longPNL.append([dt, longCF, self.longPos, 0, 0])
                else:
                    dtNext = datetime.strptime(csvfile.loc[index + 1, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
                    if endDatetime < dtNext:
                        shortPNL.append([dt, shortCF, self.shortPos, 0, 0])
                        longPNL.append([dt, longCF, self.longPos, 0, 0])
                        break
                    if dt.time() < time(18) <= dtNext.time():
                        totalResult.append(
                            [dt, self.shortPos, self.longPos, self.netPos, shortpnl, longpnl, shortreturn, longreturn, accumulateReturn])
                        shortpnl, longpnl, shortreturn, longreturn = None, None, None, None

        '''
        # 1
        targetHighLevels, targetLowLevels = [], []
        for i in range(roundForLongAndShort):
            highLevel = 'target high level %d' % (i + 1)
            lowLevel = 'target low level %d' % (i + 1)
            targetHighLevels.append(highLevel)
            targetLowLevels.append(lowLevel)

        shortInfoFileTitle = ['Date', 'high price', 'low price'] + targetHighLevels + \
                             ['order size', 'accumulated short position', 'short exit price']
        longInfoFileTitle = ['Date', 'high price', 'low price'] + targetLowLevels + \
                             ['order size', 'accumulated short position', 'long exit price']

        # 1
        shortInfofile = pd.DataFrame(shortInfo, columns=shortInfoFileTitle)
        shortInfofile.to_csv(self.resultPath + "5min_tradeinfo_short.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 2
        shortTradefile = pd.DataFrame(shortTrade, columns=['date', 'price', 'order'])
        shortTradefile.to_csv(self.resultPath + "5min_tradelog_short.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 3
        sumOfShortPNL = 0
        for shortPNLEntry in shortPNL:
            sumOfShortPNL += shortPNLEntry[-1]
        shortPNL.append(['Sum', sumOfShortPNL])
        shortpnlfile = pd.DataFrame(shortPNL, columns=['date', 'short cashflow', 'exit order', 'exit price', 'return'])
        shortpnlfile.to_csv(self.resultPath + "5min_pnl_short.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 4
        longInfofile = pd.DataFrame(longInfo, columns=longInfoFileTitle)
        longInfofile.to_csv(self.resultPath + "5min_tradeinfo_long.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 5
        longTradefile = pd.DataFrame(longTrade, columns=['date', 'price', 'order'])
        longTradefile.to_csv(self.resultPath + "5min_tradelog_long.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 6
        sumOfLongPNL = 0
        for longPNLEntry in longPNL:
            sumOfLongPNL += longPNLEntry[-1]
        longPNL.append(['Sum', sumOfLongPNL])
        longpnlfile = pd.DataFrame(longPNL, columns=['date', 'long cashflow', 'exit order', 'exit price', 'return'])
        longpnlfile.to_csv(self.resultPath + "5min_pnl_long.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # 7
        totalpnl = pd.DataFrame(totalResult,
                                columns=['date', 'short position', 'long position', 'net position', 'short pnl',
                                         'long pnl', 'short return', 'long return', 'accumulate return'])
        totalpnl.to_csv(self.resultPath + "5min_totalreturn.csv", date_format="%Y-%m-%d %H:%M:%S", index=False)

        # print some results
        print 'total return for short is %.2f' % (sumOfShortPNL * 100) + '%'
        print 'total return for long is %.2f' % (sumOfLongPNL * 100) + '%'
        '''
        print 'total return is %.2f' % (accumulateReturn * 100) + '%\n'
        return accumulateReturn

    def run(self):
        if not self.prepareDirectory():
            return None
        return self.readMarket()


def readMatsuba(filename):
    mat_dateList, mat_value = [], {}
    csvfile = pd.read_csv(filename)
    for i in csvfile.index:
        dt = datetime.strptime(csvfile.loc[i, 'DATE'], '%Y-%m-%d').date()
        mat_dateList.append(dt)
        mat_value[dt] = [csvfile.loc[i, 'HIGH'], csvfile.loc[i, 'LOW']]
    return mat_dateList, mat_value


if __name__ == "__main__":
    programStartTime = timeToCount.time()
    # convertTime(marketFile)   # convert time in excel bloomberg to local time and generate new market data file
    # ----------------------Parameters-----------------------
    poslimit = 600
    capital = 5000.0 * poslimit
    matFile = './Data/CL1 COMDTY_res2.csv'
    mat_dateList, mat_value = readMatsuba(matFile)
    startdate = datetime(2016, 1, 1).date()
    enddate = datetime(2016, 5, 30).date()

    print "start date: " + startdate.strftime('%Y_%m_%d')
    print "end data: " + enddate.strftime('%Y_%m_%d')

    amount = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 512, 11: 1024}
    unit = 10
    roundForLongAndShort = 4
    percentForALevel = 0.03
    takeProfit = 0.03

    strategy = Strategy(poslimit, capital, takeProfit, startdate, enddate, mat_dateList, mat_value)
    # strategy.run(0.03)

    percentForALevelList, roundForLongAndShortList, takeProfitList = [], [], []
    bestPercentForALevel, bestRoundForLongAndShort, bestTakeProfit, bestReturn = 0.0, 0, 0.0, 0.0

    levelStep = 0.005
    lowerBoundForLevel = 0.02
    upperBoundForLevel = 0.08
    for i in range(int((upperBoundForLevel - lowerBoundForLevel) / levelStep) + 1):
        percentForALevelList.append(lowerBoundForLevel + i * levelStep)

    takeProfitStep = 0.005
    lowerBoundFortakeProfit = 0.03
    upperBoundFortakeProfit = 0.08
    for i in range(int((upperBoundFortakeProfit - lowerBoundFortakeProfit) / takeProfitStep) + 1):
        takeProfitList.append(lowerBoundFortakeProfit + i * takeProfitStep)

    for i in range(2, len(amount) + 1):
        roundForLongAndShortList.append(i)

    for tempLevel in percentForALevelList:
        percentForALevel = tempLevel
        for tempTakeProfit in takeProfitList:
            takeProfit = tempTakeProfit
            for tempRoundForLongAndShort in roundForLongAndShortList:
                roundForLongAndShort = tempRoundForLongAndShort
                tempReturn = strategy.run()
                if bestReturn < tempReturn:
                    bestPercentForALevel = tempLevel
                    bestTakeProfit = tempTakeProfit
                    bestRoundForLongAndShort = tempRoundForLongAndShort
                    bestReturn = tempReturn

    print 'best return is %f, percent for a level is %f, take profit at %f, and round limit is %d' \
          % (bestReturn, bestPercentForALevel, bestTakeProfit, bestRoundForLongAndShort)



    #print 'best level is %f, the return is %f' % (levelForMaxReturn, maxReturn)

    # result = strategy.run(12)
    # print 'result is %.2f' % result
    # x0 = [0.03]
    # bods = [(0.02, 0.08)]
    # res = minimize(strategy.run, x0, bounds=bods, method='SLSQP')
    # res = minimize(strategy.run, x0, bounds=bods)
    # print res

    '''
    threads = []
    task = Strategy(poslimit, capital, takeProfit, startdate, enddate, mat_dateList, mat_value)
    threads.append(task)

    if len(threads) > 0:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        print "no date starts."
    '''

    print ('This program take %.2s seconds to run' % (timeToCount.time() - programStartTime))
