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
_poslimit = 600
_capital = 5000.0 * _poslimit
_sequenceForPosition = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 512, 11: 1024}
_unit = 10
_roundLimit = 7
_takeProfit = 0.025
_positionLevel = 0.02
_matFilePath = './Data/CL1 COMDTY_res2_2015-12-31_2016-06-17.csv'
_marketDataFilePath = './Data/CL1 COMDTY_2016-12-31_2016-06-19_5Minutes_simplied.csv'
_cl1ConstantsPckl = './CL1_Constants.pckl'
_cl1VariablesPckl = './CL1_Variables.pckl'


def getOpenDateForADatetime(dt):
    if time(18) <= dt.time():
        return dt.date()
    else:
        return (dt - timedelta(1)).date()

def readMatsuba(filename):
    mat_dateList, mat_value = [], {}
    csvfile = pd.read_csv(filename)
    for i in csvfile.index:
        dt = datetime.strptime(csvfile.loc[i, 'DATE'], '%Y-%m-%d').date()
        mat_dateList.append(dt)
        mat_value[dt] = [csvfile.loc[i, 'HIGH'], csvfile.loc[i, 'LOW']]
    return mat_dateList, mat_value

# todo: change this method to get the updated matsuba value
def getMatValue(dt, matPath, type):
    # Step1 get the open date for this market data
    openDate = getOpenDateForADatetime(dt)

    # Step2 check if there is a mat value of that day
    matDateList, matValueDic = readMatsuba(matPath)
    if openDate in matDateList:
        if type == 'high':
            return matValueDic[openDate][0]
        elif type == 'low':
            return matValueDic[openDate][1]
        else:
            raise ValueError('invalid type to get matsuba value')
    else:
        return None

def resetTargetList(matValue, type, roundLimit, positionLevel, sequenceForPosition, unit):
    if matValue is None:
        return [[None, 0] for i in range(roundLimit)]
    targetList = []
    for i in xrange(roundLimit):
        if type == 'high':
            targetList.append(
                [matValue * np.power((1 + positionLevel), i), -sequenceForPosition[i] * unit])
        elif type == 'low':
            targetList.append(
                [matValue * np.power((1 - positionLevel), i), sequenceForPosition[i] * unit])
        else:
            raise ValueError('invalid type to reset target list')

    return targetList


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

    def prepareDirectory(self):  # prepare both database and backup folders
        self.resultPath = './PaperTrading'
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

    def calculateTakeProfitPrice(self, position, type):
        if position == 0:
            return None
        if type == 'short':
            return abs(self.shortCashFlow / position) * (1 - self.takeProfit)
        elif type == 'long':
            return abs(self.longCashFlow / position) * (1 + self.takeProfit)
        else:
            raise ValueError('type is invalid')

    def resetMatValueAndTargetList(self, type, baseTarget=0):
        # openDate = getOpenDateForADatetime(self.dt)
        matHigh = getMatValue(dt, self.matPath, 'high')
        matLow = getMatValue(dt, self.matPath, 'low')
        if type == 'high':
            if matHigh:
                matValue = matHigh
                targetList = resetTargetList(matValue, 'high', self.roundLimit, self.positionLevel,
                                             self.sequenceForPosition, self.unit)
                if baseTarget != 0:
                    while targetList[0][0] <= baseTarget:
                        targetList = resetTargetList(targetList[0][0] * (1 + self.positionLevel), 'high',
                                                     self.roundLimit, self.positionLevel, self.sequenceForPosition, self.unit)
                return matValue, targetList
            else:
                return None, [[None, 0] for i in range(self.roundLimit)]
        elif type == 'low':
            if matLow:
                matValue = matLow
                targetList = resetTargetList(matValue, 'low', self.roundLimit, self.positionLevel,
                                             self.sequenceForPosition, self.unit)
                if baseTarget != 0:
                    while targetList[0][0] >= baseTarget:
                        targetList = resetTargetList(targetList[0][0] * (1 - self.positionLevel), 'low',
                                                     self.roundLimit, self.positionLevel, self.sequenceForPosition, self.unit)
                return matValue, targetList
            else:
                return None, [[None, 0] for i in range(self.roundLimit)]
        else:
            return None, [[None, 0] for i in range(self.roundLimit)]

    def getExerciseList(self, targetList):
        if targetList[-1][0] is None:
            return [], False

        exerciseList = []
        stopLoss = False
        # if the candle covers the target, the add teh target to execise list or exit
        for index, [price, size] in enumerate(targetList):
            if price and self.lowPrice <= price <= self.highPrice:
                # if the price hits the last level, then stop loss
                if index == len(targetList) - 1:
                    stopLoss = True
                    return [], stopLoss
                else:
                    exerciseList.append([price, size])
                    targetList[index] = [None, 0]

        return exerciseList, stopLoss

    def exercise(self, exerciseList, position, cashFlow, type):
        for price, size in exerciseList:
            # hit the limit position, set a new size
            if abs(size + self.netPosition) > self.poslimit:
                if type == 'short':
                    size = -(self.poslimit - abs(self.netPosition))
                elif type == 'long':
                    size = self.poslimit - self.netPosition
                else:
                    raise ValueError('type is invalid')

                if size == 0:
                    continue
                print type + ' exercise hits the position limit and the new size is %d' % size

            position += size
            self.netPosition += size
            cashFlow += 0 - price * size
            # infoList[-1][-3] += size
            # tradeList.append([self.dt, price, size])
            print type + ' exercise at $%.2f for %d position on ' % (price, size) + self.dt.strftime(
                '%Y-%m-%d %H:%M')

        return position, cashFlow

    def exitPosition(self, type, exitPrice, position, cashflow):
        exitorder = 0 - position
        position = 0
        self.netPosition += exitorder
        pnl = cashflow - exitorder * exitPrice
        exitReturn = 1000 * pnl / self.capital
        self.accumulateReturn += exitReturn
        cashflow = 0.0
        print type + ' exit at $%.2f, return is %.2f%% on ' % (exitPrice, exitReturn * 100) + self.dt.strftime(
            '%Y-%m-%d %H:%M')
        return position, pnl, exitReturn, cashflow

    def run(self):
        # there is no mat value
        if self.firstTargetHigh is None and self.firstTargetLow is None:
            return False

        # calculate take profit price
        shortTakeProfitPrice = self.calculateTakeProfitPrice(self.shortPos, 'short')
        longTakeProfitPrice = self.calculateTakeProfitPrice(self.longPos, 'long')

        # check if short exercise or long exercise
        shortExerciseList, shortStopLoss = self.getExerciseList(self.targetHighList)
        longExerciseList, longStopLoss = self.getExerciseList(self.targetLowList)

        # short exercise
        if len(shortExerciseList) != 0 and shortStopLoss is False:
            self.shortPos, self.shortCashFlow = self.exercise(shortExerciseList, self.shortPos, self.shortCashFlow, 'short')
            shortTakeProfitPrice = self.calculateTakeProfitPrice(self.shortPos, 'short')

        # long exercise
        if len(longExerciseList) != 0 and longStopLoss is False:
            self.longPos, self.longCashFlow = self.exercise(longExerciseList, self.longPos, self.longCashFlow, 'long')
            longTakeProfitPrice = self.calculateTakeProfitPrice(self.longPos, 'long')

        # short exit
        if (shortTakeProfitPrice and self.lowPrice <= shortTakeProfitPrice) or shortStopLoss is True:
            if self.lowPrice <= shortTakeProfitPrice:
                exitPrice = shortTakeProfitPrice
            elif shortStopLoss is True:
                exitPrice = self.lowPrice

            self.shortPos, shortpnl, shortReturn, self.shortCashFlow = \
                self.exitPosition('short', exitPrice, self.shortPos, self.shortCashFlow)
            self.firstTargetHigh, self.targetHighList = self.resetMatValueAndTargetList('high', highPriceForDt)

        # long exit
        if (longTakeProfitPrice and longTakeProfitPrice <= self.highPrice) or longStopLoss is True:
            if longTakeProfitPrice <= self.highPrice:
                exitPrice = longTakeProfitPrice
            elif longStopLoss is True:
                exitPrice = self.highPrice

            self.longPos, longpnl, longReturn, self.longCashFlow = \
                self.exitPosition('long', exitPrice, self.longPos, self.longCashFlow)
            self.firstTargetLow, self.targetLowList = self.resetMatValueAndTargetList('low', lowPriceForDt)

        return True


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
            # todo change matsuba to current matsuba value
            if firstTargetHigh is None:
                firstTargetHigh = getMatValue(dt, matFilePath, 'high')
                targetHighList = resetTargetList(firstTargetHigh, 'high', roundLimit, positionLevel, sequenceForPosition, unit)

            if firstTargetLow is None:
                firstTargetLow = getMatValue(dt, matFilePath, 'low')
                targetLowList = resetTargetList(firstTargetLow, 'low', roundLimit, positionLevel, sequenceForPosition, unit)
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
        firstTargetHigh = getMatValue(dt, matFilePath, 'high')
        firstTargetLow = getMatValue(dt, matFilePath, 'low')
        targetHighList = resetTargetList(firstTargetHigh, 'high', roundLimit, positionLevel, sequenceForPosition, unit)
        targetLowList = resetTargetList(firstTargetLow, 'low', roundLimit, positionLevel, sequenceForPosition, unit)
        '''
        if firstTargetHigh and firstTargetLow:
            print 'target high starts at %.2f, target low starts at %.2f' % (firstTargetHigh, firstTargetLow)
        else:
            print 'There is no target now' '''
        with open(_cl1VariablesPckl, 'w') as f:
            pickle.dump([accumulateReturn, shortPos, longPos, shortCashFlow, longCashFlow,
                         firstTargetHigh, firstTargetLow, targetHighList, targetLowList], f)

    # Step2: use the target and market data to run the strategy
    strategy = Strategy(dt, poslimit, accumulateReturn, shortCashFlow, longCashFlow, shortPos, longPos,
                        capital, matFilePath, sequenceForPosition, unit, roundLimit, takeProfit, positionLevel,
                        highPriceForDt, lowPriceForDt, firstTargetHigh, firstTargetLow, targetHighList, targetLowList)
    if strategy.run():
        # Step3: write the data back
        with open('./CL1_Variables.pckl', 'w') as f:
            pickle.dump([strategy.accumulateReturn, strategy.shortPos, strategy.longPos, strategy.shortCashFlow, strategy.longCashFlow
                            , strategy.firstTargetHigh, strategy.firstTargetLow, strategy.targetHighList, strategy.targetLowList], f)


if __name__ == '__main__':

    startDate = datetime(2016, 1, 1, 0, 0, 0)
    endDate = datetime(2016, 5, 31, 16, 55, 0)

    timeToTest = int((endDate - startDate).total_seconds() / timedelta(minutes=5).total_seconds())

    # todo: change this to get the martket data
    marketDataFilePath = './Data/CL1 COMDTY_2016-12-31_2016-06-19_5Minutes_simplied.csv'
    csvfile = pd.read_csv(marketDataFilePath)

    for dataIndex in csvfile.index:
        dt = datetime.strptime(csvfile.loc[dataIndex, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
        if dt < startDate:
            continue
        elif endDate < dt:
            break
        elif startDate <= dt <= endDate:
            highPriceForDt, lowPriceForDt = csvfile.loc[dataIndex, 'HIGH'], csvfile.loc[dataIndex, 'LOW']
            start(dt, highPriceForDt, lowPriceForDt)
