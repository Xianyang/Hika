import pandas as pd
from scipy.optimize import minimize
import numpy as np
import os
from datetime import datetime, time, timedelta
import pickle
import csv
import xlsxwriter
import threading
import time as timeToCount


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

def getMatValue(dt, matPath, type):
    # Step1 get the open date for this market data
    # Step2 check if there is a mat value of that day
    openDate = getOpenDateForADatetime(dt)
    matDateList, matValueDic = readMatsuba(matPath)
    if openDate in matDateList:
        if type == 'high':
            return matValueDic[openDate][0]
        elif type == 'low':
            return matValueDic[openDate][1]
        else:
            raise ValueError('invalid type')
    else:
        return None

def resetTargetList(matValue, type, roundLimit, positionLevel, sequenceForPosition, unit):
    if matValue is None:
        return None, [[None, 0] for i in range(roundLimit)]
    targetList = []
    for i in xrange(roundLimit):
        if type == 'high':
            targetList.append(
                [matValue * np.power((1 + positionLevel), i), -sequenceForPosition[i] * unit])
        elif type == 'low':
            targetList.append(
                [matValue * np.power((1 - positionLevel), i), sequenceForPosition[i] * unit])
        else:
            raise ValueError('type is invalid')

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

    def resetTargetList(self, matValue, type):
        targetList = []
        for i in xrange(self.roundLimit):
            if type == 'high':
                targetList.append([matValue * np.power((1 + self.positionLevel), i), -self.sequenceForPosition[i] * self.unit])
            elif type == 'low':
                targetList.append([matValue * np.power((1 - self.positionLevel), i), self.sequenceForPosition[i] * self.unit])
            else:
                raise ValueError('type is invalid')

        return targetList

    def calculateTakeProfitPrice(self, position, type):
        if position == 0:
            return None
        if type == 'short':
            return abs(self.shortCashFlow / position) * (1 - self.takeProfit)
        elif type == 'long':
            return abs(self.longCashFlow / position) * (1 + self.takeProfit)
        else:
            raise ValueError('type is invalid')

    def resetMatValue(self, type, baseTarget=0):
        # openDate = getOpenDateForADatetime(self.dt)
        matHigh = getMatValue(dt, self.matPath, 'high')
        matLow = getMatValue(dt, self.matPath, 'low')
        if matHigh and matLow:
            if type == 'high':
                matValue = matHigh
                targetList = self.resetTargetList(matValue, 'high')
                if baseTarget != 0:
                    while targetList[0][0] <= baseTarget:
                        targetList = self.resetTargetList(targetList[0][0] * (1 + self.positionLevel), 'high')
            elif type == 'low':
                matValue = matLow
                targetList = self.resetTargetList(matValue, 'low')
                if baseTarget != 0:
                    while targetList[0][0] >= baseTarget:
                        targetList = self.resetTargetList(targetList[0][0] * (1 - self.positionLevel))
            else:
                raise ValueError('type is invalid')

            return matValue, targetList
        else:
            return None, [[None, 0] for i in range(self.roundLimit)]

    def getExerciseList(self, targetList):
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
        if self.firstTargetHigh is None or self.firstTargetLow is None:
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
            self.firstTargetHigh, self.targetHighList = self.resetMatValue('high', highPriceForDt)

        # long exit
        if (longTakeProfitPrice and longTakeProfitPrice <= self.highPrice) or longStopLoss is True:
            if longTakeProfitPrice <= self.highPrice:
                exitPrice = longTakeProfitPrice
            elif longStopLoss is True:
                exitPrice = self.highPrice

            self.longPos, longpnl, longReturn, self.longCashFlow = \
                self.exitPosition('long', exitPrice, self.longPos, self.longCashFlow)
            self.firstTargetLow, self.targetLowList = self.resetMatValue('low', lowPriceForDt)

        return True



def start(dt, highPriceForDt, lowPriceForDt):
    # Step1 read saved data
    # print '\nstart reading saved data'
    # print 'the time is ' + dt.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open('./CL1_Constants.pckl') as f:
            poslimit, capital, sequenceForPosition, unit, roundLimit, takeProfit, positionLevel, matFilePath, marketDataFilePath = pickle.load(f)
            # print 'position limit is %d, capital is %d' % (poslimit, capital)
            # print 'unit is %d, round limit %d, take profit at %.2f%%, position level is %.2f%%' % (
            # unit, roundLimit, takeProfit * 100, positionLevel * 100)
        with open('CL1_Variables.pckl') as f:
            accumulateReturn, shortPos, longPos, shortCashFlow, longCashFlow, firstTargetHigh, firstTargetLow, targetHighList, targetLowList = pickle.load(f)
            # print 'short position is now at %d, long position is now at %d' % (shortPos, longPos)
            # todo change this time to corresponding time zone
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
        poslimit = 600
        capital = 5000.0 * poslimit
        sequenceForPosition = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 512, 11: 1024}
        unit = 10
        roundLimit = 7
        takeProfit = 0.025
        positionLevel = 0.02
        matFilePath = './Data/CL1 COMDTY_res2_2015-12-31_2016-06-17.csv'
        marketDataFilePath = './Data/CL1 COMDTY_2016-12-31_2016-06-19_5Minutes_simplied.csv'
        with open('./CL1_Constants.pckl', 'w') as f:
            pickle.dump(
                [poslimit, capital, sequenceForPosition, unit, roundLimit, takeProfit, positionLevel, matFilePath, marketDataFilePath], f)

        # set the variables
        accumulateReturn = 0.0
        shortPos, longPos = 0, 0
        shortCashFlow, longCashFlow = 0, 0
        # todo change this time to corresponding time zone
        firstTargetHigh = getMatValue(dt, matFilePath, 'high')
        firstTargetLow = getMatValue(dt, matFilePath, 'low')
        targetHighList = resetTargetList(firstTargetHigh, 'high', roundLimit, positionLevel, sequenceForPosition, unit)
        targetLowList = resetTargetList(firstTargetLow, 'low', roundLimit, positionLevel, sequenceForPosition, unit)
        '''
        if firstTargetHigh and firstTargetLow:
            print 'target high starts at %.2f, target low starts at %.2f' % (firstTargetHigh, firstTargetLow)
        else:
            print 'There is no target now' '''
        with open('./CL1_Variables.pckl', 'w') as f:
            pickle.dump([accumulateReturn, shortPos, longPos, shortCashFlow, longCashFlow,
                         firstTargetHigh, firstTargetLow, targetHighList, targetLowList], f)

    # Step2: use the target and market data to run the strategy
    strategy = Strategy(dt, poslimit, accumulateReturn, shortCashFlow, longCashFlow, shortPos, longPos, capital, matFilePath, sequenceForPosition,
                       unit, roundLimit, takeProfit, positionLevel, highPriceForDt, lowPriceForDt, firstTargetHigh, firstTargetLow, targetHighList, targetLowList)
    if strategy.run():
        # Step3: write the data back
        with open('./CL1_Variables.pckl', 'w') as f:
            pickle.dump([strategy.accumulateReturn, strategy.shortPos, strategy.longPos, strategy.shortCashFlow, strategy.longCashFlow
                            , strategy.firstTargetHigh, strategy.firstTargetLow, strategy.targetHighList, strategy.targetLowList], f)


if __name__ == '__main__':

    startDate = datetime(2016, 1, 1, 0, 0, 0)
    endDate = datetime(2016, 5, 31, 16, 55, 0)

    timeToTest = int((endDate - startDate).total_seconds() / timedelta(minutes=5).total_seconds())

    marketDataFilePath = './Data/CL1 COMDTY_2016-12-31_2016-06-19_5Minutes.csv'
    csvfile = pd.read_csv(marketDataFilePath)
    dateList = []
    for DataIndex in csvfile.index:
        dt = datetime.strptime(csvfile.loc[DataIndex, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
        dateList.append(dt)

    for i in xrange(timeToTest):
        dt = startDate + i * timedelta(minutes=5)
        if startDate <= dt <= endDate and dt in dateList:
            index = dateList.index(dt)
            highPriceForDt, lowPriceForDt = csvfile.loc[index, 'HIGH'], csvfile.loc[index, 'LOW']
            start(dt, highPriceForDt, lowPriceForDt)

