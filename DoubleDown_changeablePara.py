import pandas as pd
from scipy.optimize import minimize
import numpy as np
import os
from datetime import datetime, time, timedelta
import xlsxwriter
import threading
import time as timeToCount


class Strategy():
    def __init__(self, poslimit, capital, startdate, enddate, mat_dateList, mat_value, marketDataFilePath):
        # threading.Thread.__init__(self)
        self.poslimit = poslimit
        self.capital = capital

        # use the start date and last date to get start date of last date of matsuba data
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
        self.matdateList = mat_dateList[sindex:eindex]
        self.matvalueList = mat_value
        self.marketFilePath = marketDataFilePath
        self.resultPath = ''

    def prepareDirectory(self, unit):  # prepare both database and backup folders
        self.resultPath = './OutputCP/Unit' + str(unit) + '_' + self.startdate.strftime(
            '%Y%m%d') + '_' + self.enddate.strftime(
            '%Y%m%d') + '/'
        if not os.path.exists(self.resultPath):
            try:
                os.makedirs(self.resultPath)
            except OSError:
                print self.resultPath + " directory could not be created!"
                return False
        return True

    def resetTargetList(self, matValue, type, unit, sequenceForPosition, roundForLongAndShort, percentForALevel):
        targetList = []
        if type == 'high':
            for i in xrange(roundForLongAndShort):
                if i == 0:
                    targetList.append([matValue, -sequenceForPosition[i] * unit])
                else:
                    targetList.append([targetList[-1][0] * (1 + percentForALevel), -sequenceForPosition[i] * unit])
        elif type == 'low':
            for i in xrange(roundForLongAndShort):
                if i == 0:
                    targetList.append([matValue, sequenceForPosition[i] * unit])
                else:
                    targetList.append([targetList[-1][0] * (1 - percentForALevel), sequenceForPosition[i] * unit])
        else:
            raise ValueError('type is invalid')
        return targetList

    def calculateTakeProfitPrice(self, cashflow, position, takeProfit, type):
        if position == 0:
            return None
        if type == 'short':
            return abs(cashflow / position) * (1 - takeProfit)
        elif type == 'long':
            return abs(cashflow / position) * (1 + takeProfit)
        else:
            raise ValueError('type is invalid')

    def getOpenDateForADatetime(self, dt):
        if time(18) <= dt.time():
            return dt.date()
        else:
            return (dt - timedelta(1)).date()
        
    def resetMatValue(self, dt, type, unit, sequenceForPosition, roundForLongAndShort, percentForALevel, baseTarget=0):
        openDate = self.getOpenDateForADatetime(dt)
        if openDate in self.matdateList:
            if type == 'high':
                matValue = self.matvalueList[openDate][0]
                targetList = self.resetTargetList(matValue, 'high', unit, sequenceForPosition, roundForLongAndShort, percentForALevel)
                if baseTarget != 0:
                    while targetList[0][0] <= baseTarget:
                        targetList = self.resetTargetList(targetList[0][0] * (1 + percentForALevel), 'high', unit,
                                                          sequenceForPosition, roundForLongAndShort, percentForALevel)
            elif type == 'low':
                matValue = self.matvalueList[openDate][1]
                targetList = self.resetTargetList(matValue, 'low', unit, sequenceForPosition, roundForLongAndShort, percentForALevel)
                if baseTarget != 0:
                    while targetList[0][0] >= baseTarget:
                        targetList = self.resetTargetList(targetList[0][0] * (1 - percentForALevel), 'low', unit,
                                                          sequenceForPosition, roundForLongAndShort, percentForALevel)
            else:
                raise ValueError('type is invalid')

            return matValue, targetList
        else:
            return None, [[None, 0] for i in range(roundForLongAndShort)]

    def getExerciseList(self, highPrice, lowPrice, targetList):
        exerciseList = []
        stopLoss = False
        # if the candle covers the target, the add teh target to execise list or exit
        for index, [price, size] in enumerate(targetList):
            if price and lowPrice <= price <= highPrice:
                # if the price hits the last level, then stop loss
                if index == len(targetList) - 1:
                    stopLoss = True
                    return [], stopLoss
                else:
                    exerciseList.append([price, size])
                    targetList[index] = [None, 0]

        return exerciseList, stopLoss

    def exercise(self, dt, exerciseList, position, netPosition, cashFlow, infoList, tradeList, type):
        for price, size in exerciseList:
            # hit the limit position, set a new size
            if abs(size + netPosition) > self.poslimit:
                if type == 'short':
                    size = -(self.poslimit - abs(netPosition))
                elif type == 'long':
                    size = self.poslimit - netPosition
                else:
                    raise ValueError('type is invalid')

                if size == 0:
                    continue
                print type + ' exercise hits the position limit and the new size is %d' % size

            position += size
            netPosition += size
            cashFlow += 0 - price * size
            infoList[-1][-3] += size
            tradeList.append([dt, price, size])
            print type + ' exercise at $%.2f for %d position on ' % (price, size) + dt.strftime(
                '%Y-%m-%d %H:%M')

        return position, netPosition, cashFlow

    def exitPosition(self, type, dt, exitPrice, position, netPos, accumulateReturn, cashflow, highPriceForDt,
                  lowPriceForDt, targetList, pnlList, infoList, tradeList):
        exitorder = 0 - position
        position = 0
        netPos += exitorder
        pnl = cashflow - exitorder * exitPrice
        exitReturn = 1000 * pnl / self.capital
        accumulateReturn += exitReturn
        pnlList.append([dt, cashflow, exitorder, exitPrice, exitReturn])
        infoList.append(
            [dt, highPriceForDt ,lowPriceForDt] + [i[0] for i in targetList] + [exitorder, position, exitPrice])
        tradeList.append([dt, exitPrice, exitorder])
        cashflow = 0.0
        print type + ' exit at $%.2f, return is %.2f' % (exitPrice, exitReturn * 100) + '% on ' + dt.strftime(
            '%Y-%m-%d %H:%M')
        return position, netPos, pnl, exitReturn, accumulateReturn, cashflow

    def run(self, unit, sequenceForPosition, roundForLongAndShort, takeProfit, percentForALevel, exitAtEnd):
        # check if round limit is valid
        if len(sequenceForPosition) < roundForLongAndShort or roundForLongAndShort < 2:
            print 'invalid round %d' % roundForLongAndShort
            return -1
        print 'the stop loss round is %d, take profit at %f and level for long and short is %f' % (
        roundForLongAndShort, takeProfit, percentForALevel)

        startDatetime = datetime.combine(self.startdate, time(18))
        endDatetime = datetime.combine(self.enddate, time(17, 15))

        # parameters
        netPos, shortPos, longPos = 0, 0, 0
        shortCashFlow, longCashFlow = 0.0, 0.0
        shortInfo, shortTrade, shortPNL = [], [], []
        longInfo, longTrade, longPNL = [], [], []
        totalResult = []
        [mat_high, mat_low] = self.matvalueList[self.startdate]
        shortpnl, longpnl, shortreturn, longreturn = None, None, None, None
        accumulateReturn = 0
        accumulateReturnWithExit = 0

        # this is a list saving five levels from the high matsuba and low matsuba
        target_high = self.resetTargetList(mat_high, 'high', unit, sequenceForPosition, roundForLongAndShort,
                                           percentForALevel)
        target_low = self.resetTargetList(mat_low, 'low', unit, sequenceForPosition, roundForLongAndShort,
                                          percentForALevel)
        csvfile = pd.read_csv(self.marketFilePath)

        for DataIndex in csvfile.index:
            dt = datetime.strptime(csvfile.loc[DataIndex, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
            if dt < startDatetime:
                continue
            elif endDatetime < dt:
                break
            elif startDatetime <= dt <= endDatetime:
                highPriceForDt = csvfile.loc[DataIndex, 'HIGH']
                lowPriceForDt = csvfile.loc[DataIndex, 'LOW']

                # calculate the take profit price for short and long
                shortTakeProfitPrice = self.calculateTakeProfitPrice(shortCashFlow, shortPos, takeProfit, 'short')
                longTakeProfitPrice = self.calculateTakeProfitPrice(longCashFlow, longPos, takeProfit, 'long')

                # it means that mat_high and mat_low does not exist at this day
                if mat_high is None:
                    mat_high, target_high = self.resetMatValue(dt, 'high', unit, sequenceForPosition, roundForLongAndShort, percentForALevel)

                if mat_low is None:
                    mat_low, target_low = self.resetMatValue(dt, 'low', unit, sequenceForPosition, roundForLongAndShort, percentForALevel)

                # check if short exercise
                shortInfo.append(
                    [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_high] + [0, shortPos, shortTakeProfitPrice])
                short_exe, shortStopLoss = self.getExerciseList(highPriceForDt, lowPriceForDt, target_high)

                # short exercise
                if len(short_exe) != 0 and shortStopLoss is False:
                    shortPos, netPos, shortCashFlow = self.exercise(dt, short_exe, shortPos, netPos, shortCashFlow, shortInfo, shortTrade, 'short')
                    shortTakeProfitPrice = self.calculateTakeProfitPrice(shortCashFlow, shortPos, takeProfit, 'short')
                    shortInfo[-1][-2] = shortPos
                    shortInfo[-1][-1] = shortTakeProfitPrice

                # check if long exercise
                longInfo.append(
                    [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_low] + [0, longPos, longTakeProfitPrice])
                long_exe, longStopLoss = self.getExerciseList(highPriceForDt, lowPriceForDt, target_low)

                # long exercise
                if len(long_exe) != 0 and longStopLoss is False:
                    longPos, netPos, longCashFlow = self.exercise(dt, long_exe, longPos, netPos, longCashFlow, longInfo, longTrade, 'long')
                    longTakeProfitPrice = self.calculateTakeProfitPrice(longCashFlow, longPos, takeProfit, 'long')
                    longInfo[-1][-2] = longPos
                    longInfo[-1][-1] = longTakeProfitPrice

                # short exit
                if (shortTakeProfitPrice and lowPriceForDt <= shortTakeProfitPrice) or shortStopLoss is True:
                    if lowPriceForDt <= shortTakeProfitPrice:
                        exitPrice = shortTakeProfitPrice
                    elif shortStopLoss is True:
                        exitPrice = lowPriceForDt

                    shortPos, netPos, shortpnl, shortreturn, accumulateReturn, shortCashFlow = \
                        self.exitPosition('short', dt, exitPrice, shortPos, netPos, accumulateReturn, shortCashFlow, highPriceForDt, lowPriceForDt, target_high,
                                  shortPNL, shortInfo, shortTrade)
                    mat_high, target_high = self.resetMatValue(dt, 'high', unit, sequenceForPosition,
                                                               roundForLongAndShort, percentForALevel, highPriceForDt)

                # long exit
                if (longTakeProfitPrice and longTakeProfitPrice <= highPriceForDt) or longStopLoss is True:
                    if longTakeProfitPrice <= highPriceForDt:
                        exitPrice = longTakeProfitPrice
                    elif longStopLoss is True:
                        exitPrice = highPriceForDt
                    longPos, netPos, longpnl, longreturn, accumulateReturn, longCashFlow = \
                        self.exitPosition('long', dt, exitPrice, longPos, netPos, accumulateReturn, longCashFlow,
                                          highPriceForDt, lowPriceForDt, target_low,
                                          longPNL, longInfo, longTrade)
                    mat_low, target_low = self.resetMatValue(dt, 'low', unit, sequenceForPosition,
                                                               roundForLongAndShort, percentForALevel, lowPriceForDt)

                # the last data
                if DataIndex == csvfile.index.values[-1]:
                    shortPNL.append([dt, shortCashFlow, shortPos, 0, 0])
                    longPNL.append([dt, longCashFlow, longPos, 0, 0])
                else:
                    dtNext = datetime.strptime(csvfile.loc[DataIndex + 1, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
                    if endDatetime < dtNext:
                        # last data and exit
                        if shortPos != 0:
                            print 'short exit at the end'
                            exitPrice = lowPriceForDt
                            shortPos, netPos, shortpnl, shortreturn, accumulateReturnWithExit, shortCashFlow = \
                                self.exitPosition('short', dt, exitPrice, shortPos, netPos, accumulateReturn,
                                                  shortCashFlow, highPriceForDt, lowPriceForDt, target_high,
                                                  shortPNL, shortInfo, shortTrade)
                            totalResult.append(
                                [dt, shortPos, longPos, netPos, shortpnl, longpnl, shortreturn, longreturn,
                                 accumulateReturnWithExit])

                        if longPos != 0:
                            print 'long exit at the end'
                            exitPrice = highPriceForDt
                            longPos, netPos, longpnl, longreturn, accumulateReturnWithExit, longCashFlow = \
                                self.exitPosition('long', dt, exitPrice, longPos, netPos, accumulateReturn,
                                                  longCashFlow,
                                                  highPriceForDt, lowPriceForDt, target_low,
                                                  longPNL, longInfo, longTrade)
                            totalResult.append(
                                [dt, shortPos, longPos, netPos, shortpnl, longpnl, shortreturn, longreturn,
                                 accumulateReturnWithExit])

                        if accumulateReturnWithExit == 0:
                            accumulateReturnWithExit = accumulateReturn

                        break
                    if dt.time() < time(18) <= dtNext.time():
                        totalResult.append(
                            [dt, shortPos, longPos, netPos, shortpnl, longpnl, shortreturn, longreturn,
                             accumulateReturn])
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
        return accumulateReturn, accumulateReturnWithExit


def readMatsuba(filename):
    mat_dateList, mat_value = [], {}
    csvfile = pd.read_csv(filename)
    for i in csvfile.index:
        dt = datetime.strptime(csvfile.loc[i, 'DATE'], '%Y-%m-%d').date()
        mat_dateList.append(dt)
        mat_value[dt] = [csvfile.loc[i, 'HIGH'], csvfile.loc[i, 'LOW']]
    return mat_dateList, mat_value


def findBestParametersBackTest(startdate, enddate, poslimit, capital, unit, mat_dateList, mat_value, marketDataFilePath, sequenceForPosition):
    strategy = Strategy(poslimit, capital, startdate, enddate, mat_dateList, mat_value, marketDataFilePath)
    percentForALevelList, roundForLongAndShortList, takeProfitList = [], [], []

    # ------set the bounds for parameters------
    lowerBoundForRound = 4

    levelStep = 0.005
    lowerBoundForLevel = 0.02
    upperBoundForLevel = 0.05

    takeProfitStep = 0.005
    lowerBoundFortakeProfit = 0.02
    upperBoundFortakeProfit = 0.05

    # ------create list for data------
    for i in range(int((upperBoundForLevel - lowerBoundForLevel) / levelStep) + 1):
        percentForALevelList.append(lowerBoundForLevel + i * levelStep)

    for i in range(int((upperBoundFortakeProfit - lowerBoundFortakeProfit) / takeProfitStep) + 1):
        takeProfitList.append(lowerBoundFortakeProfit + i * takeProfitStep)

    for i in range(lowerBoundForRound, len(sequenceForPosition) + 1):
        roundForLongAndShortList.append(i)

    # ------create a xlsx file------
    workbook = xlsxwriter.Workbook('return_for_double_down.xlsx')
    percentAndBold = workbook.add_format({'num_format': '#.#0%', 'bold': True})
    percent = workbook.add_format({'num_format': '#.#0%'})
    bold = workbook.add_format({'bold': True})
    percentAndRed = workbook.add_format({'font_color': 'red', 'num_format': '#.#0%'})
    percentAndGreen = workbook.add_format({'font_color': 'green', 'num_format': '#.#0%'})
    percentRedBgYellow = workbook.add_format({'font_color': 'red', 'num_format': '#.#0%', 'bg_color': 'yellow'})
    percentGreenBgYellow = workbook.add_format({'font_color': 'green', 'num_format': '#.#0%', 'bg_color': 'yellow'})
    formatForTitle = workbook.add_format({'diag_type': 2, 'text_wrap': True})

    # start the loop to test parameters
    for tempRoundForLongAndShort in roundForLongAndShortList:
        # add a worksheet for each round limit
        worksheetName = 'Round limit %d' % tempRoundForLongAndShort
        worksheet = workbook.add_worksheet(worksheetName)
        bestReturn, worstReturn, rowBestReturn, colBestReturn, rowWorstReturn, colWorstReturn = float('-inf'), float(
            'inf'), 0, 0, 0, 0

        # set title for this worksheet
        worksheet.set_column(0, 0, 22)
        worksheet.set_row(0, 30)
        worksheet.write(0, 0, '                       position level\ntake profit', formatForTitle)

        # write the third row to worksheet(position level)
        for colIndex, value in enumerate(percentForALevelList):
            worksheet.write(0, colIndex + 1, value, percentAndBold)

        for rowIndex, tempTakeProfit in enumerate(takeProfitList):
            # write the value of take profit
            worksheet.write(rowIndex + 1, 0, tempTakeProfit, percentAndBold)
            for colIndex, tempLevel in enumerate(percentForALevelList):
                tempReturn = strategy.run(unit, sequenceForPosition, tempRoundForLongAndShort, tempTakeProfit,
                                          tempLevel, exitAtEnd=True)

                # check if best or worst
                if bestReturn < tempReturn:
                    bestReturn = tempReturn
                    rowBestReturn = rowIndex + 1
                    colBestReturn = colIndex + 1

                if tempReturn < worstReturn:
                    worstReturn = tempReturn
                    rowWorstReturn = rowIndex + 1
                    colWorstReturn = colIndex + 1

                if tempReturn < 0:
                    worksheet.write(rowIndex + 1, colIndex + 1, tempReturn, percentAndRed)
                elif tempReturn >= 0:
                    worksheet.write(rowIndex + 1, colIndex + 1, tempReturn, percentAndGreen)

        # set font color for best and worst return
        worksheet.write(rowBestReturn, colBestReturn, bestReturn, percentGreenBgYellow)
        worksheet.write(rowWorstReturn, colWorstReturn, worstReturn, percentRedBgYellow)

    workbook.close()


def threeMonthBackTest(poslimit, capital, unit, mat_dateList, mat_value, marketDataFilePath, sequenceForPosition):
    # set the date to test
    beginOfData = datetime(2016, 1, 1).date()
    endOfData = datetime(2016, 6, 17).date()
    # endOfData = datetime(2016, 4, 4).date()

    daysToTest = (endOfData - beginOfData).days - 30 * 3 - 1

    if daysToTest <= 0:
        return None

    workbook = xlsxwriter.Workbook('return_for_different_start_date.xlsx')
    bold = workbook.add_format({'bold': True})
    percentAndRed = workbook.add_format({'font_color': 'red', 'num_format': '#.##0%'})
    percentAndGreen = workbook.add_format({'font_color': 'green', 'num_format': '#.##0%'})
    percentWithoutColor = workbook.add_format({'num_format': '#0.##0%'})
    worksheet = workbook.add_worksheet('Sheet1')
    worksheet.set_column(0, 0, 10)
    worksheet.set_column(0, 1, 25)
    worksheet.set_column(0, 2, 25)
    worksheet.set_column(0, 3, 25)
    worksheet.write(0, 0, 'Start Date', bold)
    worksheet.write(0, 1, 'End Date', bold)
    worksheet.write(0, 2, 'Return(not exit at the end)', bold)
    worksheet.write(0, 3, 'Return(exit at the end)', bold)
    worksheet.write(0, 4, 'Delta', bold)

    for i in range(daysToTest):
        startdate = timedelta(days=i) + beginOfData
        enddate = timedelta(days=90) + startdate

        print "start date: " + startdate.strftime('%Y-%m-%d')
        print "end data: " + enddate.strftime('%Y-%m-%d')

        strategy = Strategy(poslimit, capital, startdate, enddate, mat_dateList, mat_value, marketDataFilePath)
        if not strategy.prepareDirectory(unit=unit):
            print 'create directory fail'
            return None

        worksheet.write(i + 1, 0, startdate.strftime('%Y-%m-%d'))
        worksheet.write(i + 1, 1, enddate.strftime('%Y-%m-%d'))

        returnNotExit, returnExit = strategy.run(unit, sequenceForPosition, roundForLongAndShort=7, takeProfit=0.025,
                                  percentForALevel=0.02, exitAtEnd=True)
        if returnNotExit < 0:
            worksheet.write(i + 1, 2, returnNotExit, percentAndRed)
        elif 0 <= returnNotExit:
            worksheet.write(i + 1, 2, returnNotExit, percentAndGreen)

        if returnExit < 0:
            worksheet.write(i + 1, 3, returnExit, percentAndRed)
        elif 0 <= returnExit:
            worksheet.write(i + 1, 3, returnExit, percentAndGreen)

        delta = returnNotExit - returnExit
        if delta < 0:
            worksheet.write(i + 1, 4, delta, percentAndRed)
        elif delta > 0:
            worksheet.write(i + 1, 4, delta, percentAndGreen)
        else:
            worksheet.write(i + 1, 4, delta, percentWithoutColor)

    workbook.close()


def startStrategy():
    poslimit = 600
    capital = 5000.0 * poslimit
    matFile = './Data/CL1 COMDTY_res2_2015-12-31_2016-06-17.csv'
    marketDataFilePath = './Data/CL1 COMDTY_2016-12-31_2016-06-19_5Minutes.csv'
    mat_dateList, mat_value = readMatsuba(matFile)
    startdate = datetime(2016, 1, 1).date()
    enddate = datetime(2016, 5, 30).date()

    sequenceForPosition = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 512, 11: 1024}
    # sequenceForPosition = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}
    unit = 10
    # these three parameters need to be changed
    # roundForLongAndShort = 4
    # percentForALevel = 0.03
    # takeProfit = 0.03

    # strategy = Strategy(poslimit, capital, startdate, enddate, mat_dateList, mat_value, marketDataFilePath)
    # strategy.run(unit, sequenceForPosition, roundForLongAndShort=4, takeProfit=0.03, percentForALevel=0.03, exitAtEnd=True)
    # strategy.run(unit, sequenceForPosition, roundForLongAndShort=7, takeProfit=0.05, percentForALevel=0.05, exitAtEnd=True)

    threeMonthBackTest(poslimit, capital, unit, mat_dateList, mat_value, marketDataFilePath, sequenceForPosition)
    # findBestParametersBackTest(startdate, enddate, poslimit, capital, unit, mat_dateList, mat_value, marketDataFilePath, sequenceForPosition)


if __name__ == "__main__":
    programStartTime = timeToCount.time()

    startStrategy()

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

    print ('This program takes %s minutes to run' % ((timeToCount.time() - programStartTime) / 60))
