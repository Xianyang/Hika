import pandas as pd
import numpy as np
import os
from datetime import datetime, time, timedelta
import pickle
import csv
import xlsxwriter
import threading
import time as timeToCount

# some constants
_cl1NewDateStartAtHour = 18
_poslimit = 600
_capital = 5000.0 * _poslimit
_sequenceForPosition = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 512, 11: 1024}
_unit = 10
_roundLimit = 7
_takeProfit = 0.025
_positionLevel = 0.02
_matFilePath = './Data/CL1 COMDTY_res2_2015-12-31_2016-06-17.csv'
_marketDataFilePath = './Data/CL1 COMDTY_2016-12-31_2016-06-19_5Minutes_simplied.csv'
_cl1ConstantsPckl = './ParaAndOutputForPaperTrading/CL1_Constants.pckl'
_cl1VariablesPckl = './ParaAndOutputForPaperTrading/CL1_Variables.pckl'
_shortTradeInfoPath = './ParaAndOutputForPaperTrading/shortTradeInfo.csv'
_longTradeInfoPath = './ParaAndOutputForPaperTrading/longTradeInfo.csv'
_shortTradeLogPath = './ParaAndOutputForPaperTrading/shortTradeLog.csv'
_longTradeLogPath = './ParaAndOutputForPaperTrading/longTradeLog.csv'

targetHighLevels, targetLowLevels = [], []
for i in range(_roundLimit):
    highLevel = 'target high level %d' % (i + 1)
    lowLevel = 'target low level %d' % (i + 1)
    targetHighLevels.append(highLevel)
    targetLowLevels.append(lowLevel)
_shortInfoFileTitle = ['Date', 'high price', 'low price'] + targetHighLevels + \
                     ['order size', 'accumulated short position', 'short exit price']
_longInfoFileTitle = ['Date', 'high price', 'low price'] + targetLowLevels + \
                    ['order size', 'accumulated short position', 'long exit price']
_tradelogTitle = ['date', 'price', 'order']


def writeOutputList(shortInfoList=[], longInfoList=[], shortTradeLogList=[], longTradeLogList=[]):
    # create 4 csv files to save trade info
    pd.DataFrame(shortInfoList, columns=_shortInfoFileTitle).to_csv(_shortTradeInfoPath, date_format="%Y-%m-%d %H:%M:%S", index=False)
    pd.DataFrame(longInfoList, columns=_longInfoFileTitle).to_csv(_longTradeInfoPath, date_format="%Y-%m-%d %H:%M:%S", index=False)
    pd.DataFrame(shortTradeLogList, columns=_tradelogTitle).to_csv(_shortTradeLogPath, date_format="%Y-%m-%d %H:%M:%S", index=False)
    pd.DataFrame(longTradeLogList, columns=_tradelogTitle).to_csv(_longTradeLogPath, date_format="%Y-%m-%d %H:%M:%S", index=False)


def loadOutputList():
    shortInfoList = pd.read_csv(_shortTradeInfoPath).values.tolist()
    longInfoList = pd.read_csv(_longTradeInfoPath).values.tolist()
    shortTradeLogList = pd.read_csv(_shortTradeLogPath).values.tolist()
    longTradeLogList = pd.read_csv(_longTradeLogPath).values.tolist()

    return shortInfoList, longInfoList, shortTradeLogList, longTradeLogList


def getOpenDateForADatetime(dt):
    if time(_cl1NewDateStartAtHour) <= dt.time():
        return dt.date()
    else:
        return (dt - timedelta(days=1)).date()


def readMatsuba(filename):
    mat_dateList, mat_value = [], {}
    csvfile = pd.read_csv(filename)
    for i in csvfile.index:
        dt = datetime.strptime(csvfile.loc[i, 'DATE'], '%Y-%m-%d').date()
        mat_dateList.append(dt)
        mat_value[dt] = [csvfile.loc[i, 'HIGH'], csvfile.loc[i, 'LOW']]
    return mat_dateList, mat_value


# todo: change this method to get the updated matsuba value
def getMatValue(dt, matPath, positionType):
    # Step1 get the open date for this market data
    openDate = getOpenDateForADatetime(dt)

    # Step2 check if there is a mat value of that day
    matDateList, matValueDic = readMatsuba(matPath)
    if openDate in matDateList:
        if positionType == 'high':
            return matValueDic[openDate][0]
        elif positionType == 'low':
            return matValueDic[openDate][1]
        else:
            raise ValueError('invalid type to get matsuba value')
    else:
        return None


def noneMatAndTargetList(roundLimit):
    return None, [[None, 0] for i in range(roundLimit)]


def resetTargetList(matValue, positionType, roundLimit, positionLevel, sequenceForPosition, unit):
    if matValue is None:
        return [[None, 0] for i in range(roundLimit)]
    targetList = []
    for i in xrange(roundLimit):
        if positionType == 'high':
            targetList.append(
                [round(matValue * np.power((1 + positionLevel), i), 2), -sequenceForPosition[i] * unit])
        elif positionType == 'low':
            targetList.append(
                [round(matValue * np.power((1 - positionLevel), i), 2), sequenceForPosition[i] * unit])
        else:
            raise ValueError('invalid type to reset target list')

    return targetList


def resetMatValueAndTargetList(positionType, matPath, roundLimit, positionLevel, sequenceForPosition, unit, baseTarget=-1):
    # openDate = getOpenDateForADatetime(dt)
    matHigh = getMatValue(dt, matPath, 'high')
    matLow = getMatValue(dt, matPath, 'low')
    if positionType == 'high':
        if matHigh:
            matValue = matHigh
            targetList = resetTargetList(matValue, 'high', roundLimit, positionLevel,
                                         sequenceForPosition, unit)
            if baseTarget != -1:
                while targetList[0][0] <= baseTarget:
                    targetList = resetTargetList(targetList[0][0] * (1 + positionLevel), 'high',
                                                 roundLimit, positionLevel, sequenceForPosition, unit)
            return matValue, targetList
        else:
            return noneMatAndTargetList(roundLimit)
    elif positionType == 'low':
        if matLow:
            matValue = matLow
            targetList = resetTargetList(matValue, 'low', roundLimit, positionLevel,
                                         sequenceForPosition, unit)
            if baseTarget != -1:
                while targetList[0][0] >= baseTarget:
                    targetList = resetTargetList(targetList[0][0] * (1 - positionLevel), 'low',
                                                 roundLimit, positionLevel, sequenceForPosition, unit)
            return matValue, targetList
        else:
            return noneMatAndTargetList(roundLimit)
    else:
        return noneMatAndTargetList(roundLimit)


class Strategy():
    def __init__(self, dt, poslimit, accumulateReturn, shortCashFlow, longCashFlow, shortPos,
                 longPos, capital, matPath, sequenceForPosition, unit, roundLimit, takeProfit,
                 positionLevel, highPrice, lowPrice, firstTargetHigh, firstTargetLow, targetHighList, targetLowList):
        self.dt = dt
        self.poslimit = poslimit
        self.accumulateReturn = accumulateReturn
        self.shortCashFlow = shortCashFlow
        self.longCashFlow = longCashFlow
        self.shortPos = shortPos
        self.longPos = longPos
        self.netPosition = shortPos + longPos
        self.capital = capital
        self.matPath = matPath
        self.sequenceForPosition = sequenceForPosition
        self.unit = unit
        self.roundLimit = roundLimit
        self.takeProfit = takeProfit
        self.positionLevel = positionLevel
        self.highPrice = highPrice
        self.lowPrice = lowPrice
        self.firstTargetHigh = firstTargetHigh
        self.firstTargetLow = firstTargetLow
        self.targetHighList = targetHighList
        self.targetLowList = targetLowList
        self.resultPath = ''

    # prepare the dic to save constants, variables and trade info
    def prepareDirectory(self):
        self.resultPath = './ParaAndOutputForPaperTrading'
        if not os.path.exists(self.resultPath):
            try:
                os.makedirs(self.resultPath)
            except OSError:
                print self.resultPath + " directory could not be created!"
                return False
        return True

    def dailyReturn(self, shortReturn, longReturn):
        if shortReturn is None:
            shortReturn = 0
        if longReturn is None:
            longReturn = 0

        return shortReturn + longReturn

    def calculateTakeProfitPrice(self, position, positionType):
        if position == 0:
            return None
        if positionType == 'short':
            return abs(self.shortCashFlow / position) * (1 - self.takeProfit)
        elif positionType == 'long':
            return abs(self.longCashFlow / position) * (1 + self.takeProfit)
        else:
            raise ValueError('invalid type to calculate take profit price')

    # check if there is short exercise or long exercise
    def getExerciseList(self, targetList, positionType):
        exerciseList = []
        stopLoss = False
        # if the candle covers the target, the add the target to execise list or exit
        for index, [price, size] in enumerate(targetList):
            if price and self.lowPrice <= price <= self.highPrice:
                # if the price hits the last level, then stop loss
                if index == len(targetList) - 1:
                    stopLoss = True
                    return [], stopLoss
                else:
                    exerciseList.append([price, size, positionType])
                    targetList[index] = [None, 0]

        return exerciseList, stopLoss

    def exercise(self, exerciseList, position, cashFlow, infoList, tradeLogList, positionType):
        for price, size, exerciseType in exerciseList:
            # if the price hits the limit position, the it should set a new size for this price
            if self.poslimit < abs(size + self.netPosition):
                if positionType == 'short':
                    size = -self.poslimit - self.netPosition
                elif positionType == 'long':
                    size = self.poslimit - self.netPosition
                else:
                    raise ValueError('invalid type to exercise')

                if size == 0:
                    continue
                print positionType + ' exercise hits the position limit and the new size is %d' % size

            position += size
            self.netPosition += size
            infoList[-1][-3] += size
            tradeLogList.append([self.dt, price, size])
            cashFlow += 0 - price * size
            print positionType + ' exercise at $%.2f for %d position on ' % (price, size) + self.dt.strftime('%Y-%m-%d %H:%M')

        return position, cashFlow

    def exitPosition(self, positionType, exitPrice, position, cashflow, targetList, infoList, tradeLogList):
        exitorder = 0 - position
        position = 0
        self.netPosition += exitorder
        pnl = cashflow - exitorder * exitPrice
        exitReturn = 1000 * pnl / self.capital
        self.accumulateReturn += exitReturn
        infoList.append(
            [self.dt, self.highPrice, self.lowPrice] + [i[0] for i in targetList] + [exitorder, position, exitPrice])
        tradeLogList.append([self.dt, exitPrice, exitorder])
        cashflow = 0.0
        print positionType + ' exit at $%.2f, return is %.2f%% on ' % (exitPrice, exitReturn * 100) + self.dt.strftime(
            '%Y-%m-%d %H:%M')
        return position, pnl, exitReturn, cashflow

    def run(self):
        # there is no mat value
        if self.firstTargetHigh is None and self.firstTargetLow is None:
            return []

        # shortInfo, longInfo, shortTradeLog, longTradeLog = loadOutputList()
        shortInfo, longInfo, shortTradeLog, longTradeLog = [], [], [], []

        # calculate take profit price
        shortTakeProfitPrice = self.calculateTakeProfitPrice(self.shortPos, 'short')
        longTakeProfitPrice = self.calculateTakeProfitPrice(self.longPos, 'long')

        # add the info to info list
        shortInfo.append([self.dt, self.highPrice, self.lowPrice] + [i[0] for i in self.targetHighList] +
                         [0, self.shortPos, shortTakeProfitPrice])
        longInfo.append([self.dt, self.highPrice, self.lowPrice] + [i[0] for i in self.targetLowList] +
                         [0, self.longPos, longTakeProfitPrice])

        # check if short exercise or long exercise
        shortExerciseList, shortStopLoss = self.getExerciseList(self.targetHighList, 'short')
        longExerciseList, longStopLoss = self.getExerciseList(self.targetLowList, 'long')
        orders = shortExerciseList + longExerciseList

        # short exercise
        if len(shortExerciseList) != 0 and shortStopLoss is False:
            self.shortPos, self.shortCashFlow = self.exercise(shortExerciseList, self.shortPos,
                                                              self.shortCashFlow, shortInfo, shortTradeLog, 'short')
            shortTakeProfitPrice = self.calculateTakeProfitPrice(self.shortPos, 'short')
            shortInfo[-1][-2] = self.shortPos
            shortInfo[-1][-1] = shortTakeProfitPrice

        # long exercise
        if len(longExerciseList) != 0 and longStopLoss is False:
            self.longPos, self.longCashFlow = self.exercise(longExerciseList, self.longPos,
                                                            self.longCashFlow, longInfo, longTradeLog, 'long')
            longTakeProfitPrice = self.calculateTakeProfitPrice(self.longPos, 'long')
            longInfo[-1][-2] = self.longPos
            longInfo[-1][-1] = longTakeProfitPrice

        # short exit
        if (shortTakeProfitPrice and self.lowPrice <= shortTakeProfitPrice) or shortStopLoss is True:
            if self.lowPrice <= shortTakeProfitPrice:
                exitPrice = shortTakeProfitPrice
            elif shortStopLoss is True:
                exitPrice = self.lowPrice

            orders.append([exitPrice, self.shortPos, 'short_exit'])
            self.shortPos, shortpnl, shortReturn, self.shortCashFlow = \
                self.exitPosition('short', exitPrice, self.shortPos, self.shortCashFlow, self.targetHighList, shortInfo, shortTradeLog)
            self.firstTargetHigh, self.targetHighList = resetMatValueAndTargetList('high', self.matPath, self.roundLimit,
                                                                                   self.positionLevel, self.sequenceForPosition, self.unit, self.highPrice)

        # long exit
        if (longTakeProfitPrice and longTakeProfitPrice <= self.highPrice) or longStopLoss is True:
            if longTakeProfitPrice <= self.highPrice:
                exitPrice = longTakeProfitPrice
            elif longStopLoss is True:
                exitPrice = self.highPrice

            orders.append([exitPrice, self.longPos, 'long_exit'])
            self.longPos, longpnl, longReturn, self.longCashFlow = \
                self.exitPosition('long', exitPrice, self.longPos, self.longCashFlow, self.targetLowList, longInfo, longTradeLog)
            self.firstTargetLow, self.targetLowList = resetMatValueAndTargetList('low', self.matPath, self.roundLimit,
                                                                                 self.positionLevel, self.sequenceForPosition, self.unit, self.lowPrice)

        # write the output list
        # writeOutputList(shortInfo, longInfo, shortTradeLog, longTradeLog)

        return orders


def start(dt, highPriceForDt, lowPriceForDt):
    # Step1 read saved data
    # print '\nstart reading saved data'
    # print 'the time is ' + dt.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(_cl1ConstantsPckl) as f:
            poslimit, capital, sequenceForPosition, unit, roundLimit, takeProfit, positionLevel, matFilePath, marketDataFilePath = pickle.load(f)
            # print 'position limit is %d, capital is %d' % (poslimit, capital)
            # print 'unit is %d, round limit %d, take profit at %.2f%%, position level is %.2f%%' % (
            # unit, roundLimit, takeProfit * 100, positionLevel * 100)
        with open(_cl1VariablesPckl) as f:
            accumulateReturn, shortPos, longPos, shortCashFlow, longCashFlow, firstTargetHigh, firstTargetLow, targetHighList, targetLowList = pickle.load(f)
            # print 'short position is now at %d, long position is now at %d' % (shortPos, longPos)
            # check matsuba value. if the value is None, then the current mat value will be requested.
            if firstTargetHigh is None:
                firstTargetHigh, targetHighList = resetMatValueAndTargetList('high', matFilePath, roundLimit,
                                                                             positionLevel, sequenceForPosition, unit, highPriceForDt)

            if firstTargetLow is None:
                firstTargetLow, targetLowList = resetMatValueAndTargetList('low', matFilePath, roundLimit,
                                                                           positionLevel, sequenceForPosition, unit, lowPriceForDt)
            # todo remove the comment of print
            '''
            if firstTargetHigh and firstTargetLow:
                print 'target high starts at %.2f, target low starts at %.2f' % (firstTargetHigh, firstTargetLow)
            else:
                print 'There is no target now' '''
    except IOError:
        print 'this is the first time running this strategy'
        # set the constants
        poslimit = _poslimit
        capital = _capital
        sequenceForPosition = _sequenceForPosition
        unit = _unit
        roundLimit = _roundLimit
        takeProfit = _takeProfit
        positionLevel = _positionLevel
        matFilePath = _matFilePath
        marketDataFilePath = _marketDataFilePath
        with open(_cl1ConstantsPckl, 'w') as f:
            pickle.dump([poslimit, capital, sequenceForPosition, unit, roundLimit, takeProfit,
                         positionLevel, matFilePath, marketDataFilePath], f)

        # set the variables
        accumulateReturn = 0.0
        shortPos, longPos = 0, 0
        shortCashFlow, longCashFlow = 0, 0
        # todo change matsuba to the start date
        firstTargetHigh, targetHighList = resetMatValueAndTargetList('high', matFilePath, roundLimit, positionLevel,
                                                                     sequenceForPosition, unit, highPriceForDt)
        firstTargetLow, targetLowList = resetMatValueAndTargetList('low', matFilePath, roundLimit, positionLevel,
                                                                   sequenceForPosition, unit, lowPriceForDt)
        # todo remove the comment of print
        '''
        if firstTargetHigh and firstTargetLow:
            print 'target high starts at %.2f, target low starts at %.2f' % (firstTargetHigh, firstTargetLow)
        else:
            print 'There is no target now' '''
        with open(_cl1VariablesPckl, 'w') as f:
            pickle.dump([accumulateReturn, shortPos, longPos, shortCashFlow, longCashFlow,
                         firstTargetHigh, firstTargetLow, targetHighList, targetLowList], f)

        # create the output csv file
        # writeOutputList()

    # Step2: use the target and market data to run the strategy
    strategy = Strategy(dt, poslimit, accumulateReturn, shortCashFlow, longCashFlow, shortPos, longPos,
                        capital, matFilePath, sequenceForPosition, unit, roundLimit, takeProfit, positionLevel,
                        highPriceForDt, lowPriceForDt, firstTargetHigh, firstTargetLow, targetHighList, targetLowList)
    orders = strategy.run()
    # if len(orders):
        # this is the return orders of the strategy, please use this orders to do paper trading
        # print orders


    # Step3: write the data back
    with open(_cl1VariablesPckl, 'w') as f:
        pickle.dump([strategy.accumulateReturn, strategy.shortPos, strategy.longPos, strategy.shortCashFlow, strategy.longCashFlow
                        , strategy.firstTargetHigh, strategy.firstTargetLow, strategy.targetHighList, strategy.targetLowList], f)


if __name__ == '__main__':
    startDate = datetime(2016, 1, 1, 0, 0, 0)
    # endDate = datetime(2016, 5, 31, 16, 55, 0)
    endDate = datetime(2016, 6, 16, 16, 55, 0)

    timeToTest = int((endDate - startDate).total_seconds() / timedelta(minutes=5).total_seconds())

    # todo: change this to get the martket data then you can call start()
    # ------------GUIDE------------
    # step1: get dt(datetime)
    # step2: get high price and low price
    # step3: use there three parameters to start the strategy---start(dt, highPriceForDt, lowPriceForDt)
    # step4: you can delete or comment the following test code
    # ---------END OF GUIDE--------
    csvfile = pd.read_csv(_marketDataFilePath)

    for dataIndex in csvfile.index:
        dt = datetime.strptime(csvfile.loc[dataIndex, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
        if dt < startDate:
            continue
        elif endDate < dt:
            break
        elif startDate <= dt <= endDate:
            highPriceForDt, lowPriceForDt = csvfile.loc[dataIndex, 'HIGH'], csvfile.loc[dataIndex, 'LOW']
            start(dt, highPriceForDt, lowPriceForDt)
