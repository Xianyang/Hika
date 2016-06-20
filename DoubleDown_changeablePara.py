import pandas as pd
from scipy.optimize import minimize
import numpy as np
import os
from datetime import datetime, time, timedelta
import xlsxwriter
import threading
import time as timeToCount


class Strategy():
    def __init__(self, poslimit, capital, startdate, enddate, mat_dateList, mat_value):
        # threading.Thread.__init__(self)
        self.poslimit = poslimit
        self.capital = capital
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
        self.marketFile = "./Data/CL1 COMDTY_2016-12-31_2016-06-19_5Minutes.csv"
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

    def resetTargetList(self, matValue, indicator, unit, sequenceForPosition, roundForLongAndShort, percentForALevel):
        targetList = []
        if indicator == 'High':
            for i in xrange(roundForLongAndShort):
                if i == 0:
                    targetList.append([matValue, -sequenceForPosition[i] * unit])
                else:
                    targetList.append([targetList[-1][0] * (1 + percentForALevel), -sequenceForPosition[i] * unit])
        elif indicator == 'Low':
            for i in xrange(roundForLongAndShort):
                if i == 0:
                    targetList.append([matValue, sequenceForPosition[i] * unit])
                else:
                    targetList.append([targetList[-1][0] * (1 - percentForALevel), sequenceForPosition[i] * unit])
        else:
            raise ValueError('indicator is invalid')
        return targetList

    def run(self, unit, sequenceForPosition, roundForLongAndShort, takeProfit, percentForALevel):

        if len(sequenceForPosition) < roundForLongAndShort or roundForLongAndShort < 2:
            print 'invalid round %d' % roundForLongAndShort
            return -1
        print 'the stop loss round is %d, take profit at %f and level for long and short is %f' % (
        roundForLongAndShort, takeProfit, percentForALevel)

        startDatetime = datetime.combine(self.startdate, time(18))
        endDatetime = datetime.combine(self.enddate, time(17, 15))

        # parameters
        netPos, shortPos, longPos = 0, 0, 0
        shortCF, longCF = 0.0, 0.0
        shortInfo, shortTrade, shortPNL = [], [], []
        longInfo, longTrade, longPNL = [], [], []
        totalResult = []
        [mat_high, mat_low] = self.matvalue[self.startdate]
        short_exe, long_exe = [], []
        shortpnl, longpnl, shortreturn, longreturn = None, None, None, None
        accumulateReturn = 0

        # this is a list saving five levels from the high matsuba and low matsuba
        target_high = self.resetTargetList(mat_high, 'High', unit, sequenceForPosition, roundForLongAndShort,
                                           percentForALevel)
        target_low = self.resetTargetList(mat_low, 'Low', unit, sequenceForPosition, roundForLongAndShort,
                                          percentForALevel)

        csvfile = pd.read_csv(self.marketFile)
        for DataIndex in csvfile.index:
            dt = datetime.strptime(csvfile.loc[DataIndex, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
            if dt < startDatetime:
                continue
            elif endDatetime < dt:
                break
            elif startDatetime <= dt <= endDatetime:
                highPriceForDt = csvfile.loc[DataIndex, 'HIGH']
                lowPriceForDt = csvfile.loc[DataIndex, 'LOW']
                shortStopLoss, longStopLoss = False, False

                # calculate the take profit price for short and long
                if shortPos != 0:
                    short_takePorfit_price = abs(shortCF / shortPos) * (1 - takeProfit)
                else:
                    short_takePorfit_price = None
                if longPos != 0:
                    long_takePorfit_price = abs(longCF / longPos) * (1 + takeProfit)
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
                        target_high = self.resetTargetList(mat_high, 'High', unit, sequenceForPosition,
                                                           roundForLongAndShort, percentForALevel)

                if mat_low is None:
                    if dt.time() >= time(18):
                        opent = dt.date()
                    else:
                        opent = (dt - timedelta(1)).date()
                    if opent in self.matdate:
                        mat_low = self.matvalue[opent][1]
                        target_low = self.resetTargetList(mat_low, 'Low', unit, sequenceForPosition,
                                                          roundForLongAndShort, percentForALevel)

                # short bias, using target_high
                # the info dic is [dt, high_value, low_value, target1, target2, target3, target4, target5, size, position, exit_price]
                shortInfo.append(
                    [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_high] + [0, shortPos,
                                                                                         short_takePorfit_price])
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
                        if abs(netPos + size) > self.poslimit:
                            size = -(self.poslimit - abs(netPos))
                            if size == 0:
                                continue
                            print 'short exercise hits the position limit and the new size is %d' % size
                        shortPos += size
                        netPos += size
                        shortCF += 0 - price * size  # shortCF > 0
                        shortInfo[-1][-3] += size
                        shortTrade.append([dt, price, size])
                        print 'short exercise at $%.2f for %d position on ' % (price, size) + dt.strftime(
                            '%Y-%m-%d %H:%M')

                    short_takePorfit_price = abs(shortCF / shortPos) * (1 - takeProfit)
                    short_exe = []
                    shortInfo[-1][-2] = shortPos
                    shortInfo[-1][-1] = short_takePorfit_price

                # long bias, using target_low
                # the info dic is [dt, high_value, low_value, target1, target2, target3, target4, target5, size, position, exit_price]
                longInfo.append(
                    [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_low] + [0, longPos,
                                                                                        long_takePorfit_price])
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
                        if abs(netPos + size) > self.poslimit:
                            size = self.poslimit - netPos
                            if size == 0:
                                continue
                            print 'long exercise hits the position limit and the new size is %d' % size
                        longPos += size
                        netPos += size
                        longCF += 0 - price * size  # longCF > 0
                        longInfo[-1][-3] += size
                        longTrade.append([dt, price, size])
                        print 'long exercise at $%.2f for %d position on ' % (price, size) + dt.strftime(
                            '%Y-%m-%d %H:%M')

                    long_takePorfit_price = abs(longCF / longPos) * (1 + takeProfit)
                    long_exe = []
                    longInfo[-1][-2] = longPos
                    longInfo[-1][-1] = long_takePorfit_price

                # short exit
                if (short_takePorfit_price and lowPriceForDt <= short_takePorfit_price) or shortStopLoss is True:
                    if lowPriceForDt <= short_takePorfit_price:
                        exitPrice = short_takePorfit_price
                    elif shortStopLoss is True:
                        exitPrice = lowPriceForDt

                    exitorder = 0 - shortPos
                    shortPos = 0
                    netPos += exitorder
                    shortpnl = shortCF - exitorder * exitPrice
                    shortreturn = 1000 * shortpnl / self.capital
                    accumulateReturn += shortreturn
                    shortPNL.append([dt, shortCF, exitorder, exitPrice, shortreturn])
                    shortInfo.append(
                        [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_high] + [exitorder, shortPos,
                                                                                             exitPrice])
                    shortTrade.append([dt, exitPrice, exitorder])
                    shortCF = 0.0
                    print 'short exit at $%.2f, return is %.2f' % (
                    exitPrice, shortreturn * 100) + '% on ' + dt.strftime('%Y-%m-%d %H:%M')
                    if dt.time() >= time(18):
                        opent = dt.date()
                    else:
                        opent = (dt - timedelta(1)).date()
                    if opent in self.matdate:
                        mat_high = self.matvalue[opent][0]
                        target_high = self.resetTargetList(mat_high, 'High', unit, sequenceForPosition,
                                                           roundForLongAndShort, percentForALevel)
                        while target_high[0][0] <= highPriceForDt:
                            target_high = self.resetTargetList(target_high[0][0] * (1 + percentForALevel), 'High', unit,
                                                               sequenceForPosition, roundForLongAndShort,
                                                               percentForALevel)
                    else:
                        mat_high = None
                        target_high = [[None, 0] for i in range(roundForLongAndShort)]

                # long exit
                if (long_takePorfit_price and long_takePorfit_price <= highPriceForDt) or longStopLoss is True:
                    if long_takePorfit_price <= highPriceForDt:
                        exitPrice = long_takePorfit_price
                    elif longStopLoss is True:
                        exitPrice = highPriceForDt

                    exitorder = 0 - longPos
                    longPos = 0
                    netPos += exitorder
                    longpnl = longCF - exitorder * exitPrice
                    longreturn = 1000 * longpnl / self.capital
                    accumulateReturn += longreturn
                    longPNL.append([dt, longCF, exitorder, exitPrice, longreturn])
                    longInfo.append(
                        [dt, highPriceForDt, lowPriceForDt] + [i[0] for i in target_low] + [exitorder, longPos,
                                                                                            exitPrice])
                    longTrade.append([dt, exitPrice, exitorder])
                    longCF = 0.0
                    print 'long exit at $%.2f, return is %.2f' % (exitPrice, longreturn * 100) + '% on ' + dt.strftime(
                        '%Y-%m-%d %H:%M')
                    if dt.time() >= time(18):
                        opent = dt.date()
                    else:
                        opent = (dt - timedelta(1)).date()
                    if opent in self.matdate:
                        mat_low = self.matvalue[opent][1]
                        target_low = self.resetTargetList(mat_low, 'Low', unit, sequenceForPosition,
                                                          roundForLongAndShort, percentForALevel)
                        while lowPriceForDt <= target_low[0][0]:
                            target_low = self.resetTargetList(target_low[0][0] * (1 - percentForALevel), 'Low', unit,
                                                              sequenceForPosition, roundForLongAndShort,
                                                              percentForALevel)
                    else:
                        mat_low = None
                        target_low = [[None, 0] for i in range(roundForLongAndShort)]

                # the last data
                if DataIndex == csvfile.index.values[-1]:
                    shortPNL.append([dt, shortCF, shortPos, 0, 0])
                    longPNL.append([dt, longCF, longPos, 0, 0])
                else:
                    dtNext = datetime.strptime(csvfile.loc[DataIndex + 1, 'Date'][0:19], '%Y-%m-%d %H:%M:%S')
                    if endDatetime < dtNext:
                        shortPNL.append([dt, shortCF, shortPos, 0, 0])
                        longPNL.append([dt, longCF, longPos, 0, 0])
                        print 'last date'
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
        return accumulateReturn


def readMatsuba(filename):
    mat_dateList, mat_value = [], {}
    csvfile = pd.read_csv(filename)
    for i in csvfile.index:
        dt = datetime.strptime(csvfile.loc[i, 'DATE'], '%Y-%m-%d').date()
        mat_dateList.append(dt)
        mat_value[dt] = [csvfile.loc[i, 'HIGH'], csvfile.loc[i, 'LOW']]
    return mat_dateList, mat_value


def startStrategy():
    poslimit = 600
    capital = 5000.0 * poslimit
    matFile = './Data/CL1 COMDTY_res2_2015-12-31_2016-06-17.csv'
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

    # strategy = Strategy(poslimit, capital, startdate, enddate, mat_dateList, mat_value)
    # strategy.run(unit, sequenceForPosition, roundForLongAndShort=7, takeProfit=0.025, percentForALevel=0.02)

    # set the date to test
    beginOfData = datetime(2016, 1, 1).date()
    endOfData = datetime(2016, 6, 17).date()

    daysToTest = (endOfData - beginOfData).days - 31 * 3

    if daysToTest <= 0:
        return None

    workbook = xlsxwriter.Workbook('return_for_different_start_date.xlsx')
    bold = workbook.add_format({'bold': True})
    percentAndRed = workbook.add_format({'font_color': 'red', 'num_format': '#.#0%'})
    percentAndGreen = workbook.add_format({'font_color': 'green', 'num_format': '#.#0%'})
    worksheet = workbook.add_worksheet('Sheet1')
    worksheet.set_column(0, 0, 12)
    worksheet.write(0, 0, 'Date', bold)
    worksheet.write(0, 1, 'Return', bold)

    for i in range(daysToTest):
        startdate = timedelta(days=i) + beginOfData
        enddate = timedelta(days=90) + startdate

        print "start date: " + startdate.strftime('%Y_%m_%d')
        print "end data: " + enddate.strftime('%Y_%m_%d')

        strategy = Strategy(poslimit, capital, startdate, enddate, mat_dateList, mat_value)
        if not strategy.prepareDirectory(unit=unit):
            print 'create directory fail'
            return None

        tempReturn = strategy.run(unit, sequenceForPosition, roundForLongAndShort=7, takeProfit=0.025, percentForALevel=0.02)
        worksheet.write(i + 1, 0, startdate.strftime('%Y_%m_%d'))
        if tempReturn < 0:
            worksheet.write(i + 1, 1, tempReturn, percentAndRed)
        elif 0 <= tempReturn:
            worksheet.write(i + 1, 1, tempReturn, percentAndGreen)

    workbook.close()

    '''
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
        bestReturn, worstReturn, rowBestReturn, colBestReturn, rowWorstReturn, colWorstReturn = float('-inf'), float('inf'), 0, 0, 0, 0

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
                tempReturn = strategy.run(unit, sequenceForPosition, tempRoundForLongAndShort, tempTakeProfit, tempLevel)

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
    '''


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
