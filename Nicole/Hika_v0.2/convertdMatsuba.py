import os, shutil
import xlrd
import datetime
from datetime import datetime, timedelta, date
from pytz import timezone
import threading
import pandas as pd
import numpy as np



#------------------------Constants-----------------------
GMT = timezone('GMT+0')
timezoneList = {'HKE': timezone('Asia/Hong_Kong'), 'NYMEX': timezone('US/Eastern'), 'TSX': timezone('Canada/Eastern'), 'ASX': timezone('Australia/Sydney'), 'FWB': timezone('CET'), 'LSE': timezone('Europe/London'), 'NYSE': timezone('US/Eastern')}

NOW = datetime.now()
TODAY = date.today()

# File Path
filePath = "M:\\Tanya\\2016_04_15(matsuba)\\"
output = "M:\\Tanya\\dailyMatsuba\\"

# Which matsuba to use, 2: res2, 3: res3
res2 = 2
res3 = 3

TICKER = ['CL1 COMDTY']

# host and port for connecting to bloomberg
HOST = "localhost"
PORT = 8194
#--------------------end of Constants---------------------


class getMatsubadaily(threading.Thread):
    def __init__(self, ticker):
        threading.Thread.__init__(self)
        writeLog("-----------------------------"+ticker+"----------------------------")
        if ticker == 'CL1 COMDTY':
        	self.name = "OIL LOG CLOSE"
        else:
        	self.name = ticker+ " LOG CLOSE"
        self.ticker = ticker
        #self.tz = timezoneList[exchange]
        #self.closingTime = closingTime[exchange]
        #self.endDate = datetime.now(timezoneList['HKE']).astimezone(self.tz).date()
        self.startDate = datetime(2015,9,21).date()
        self.endDate = TODAY

        self.path = output+self.ticker+'\\'   # output path
        self.file = [self.ticker+'_res2.csv', self.ticker+'_res3.csv']    # output filename

    def prepareDirectory(self):   # prepare both database and backup folders
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print self.path+" directory could not be created"
                writeLog(self.path+" directory could not be created")
                return False
        return True

    def getMatsuba(self):
        self.dateList = []
        self.res2 = []
        self.res3 = []
        dt = self.startDate
        while dt <= self.endDate:
            if os.path.exists(filePath+dt.strftime("%Y_%m_%d")):   # folder exists?
                fileList = os.listdir(filePath+dt.strftime("%Y_%m_%d"))    # get all files
                for name in fileList:
                    if name[0:len(self.name)]==self.name[0:len(self.name)] and len(name) < len(self.name)+21 and len(name) > len(self.name)+5:   # get the specific file
                        writeLog("Read: "+dt.strftime("%Y_%m_%d")+'\\'+name)
                        print "Read: "+dt.strftime("%Y_%m_%d")+'\\'+name
                        
                        book = xlrd.open_workbook(filePath+dt.strftime("%Y_%m_%d")+'\\'+name)
                        sheet = book.sheet_by_index(0)
                        i = 0
                        while sheet.cell_type(i,0) != 0:
                            i+=1
                        date = sheet.cell(i-1,0).value
                        DATE = datetime(*xlrd.xldate_as_tuple(date, 0))
                        if datetime.weekday(DATE) == 4:
                            DATE = DATE+timedelta(2)
                        DATE = pd.to_datetime(DATE, format='%Y-%m-%d')
                        if DATE in self.dateList:
                            pos = self.dateList.index(DATE)
                            if self.res2[pos][0] == DATE:
                                self.res2[pos] = (DATE, sheet.cell(i,19).value, sheet.cell(i,22).value)
                            if self.res3[pos][0] == DATE:
                                self.res3[pos] = (DATE, sheet.cell(i,20).value, sheet.cell(i,23).value)
                        else:
                            self.dateList.append(DATE)
                            self.res2.append((DATE, sheet.cell(i,19).value, sheet.cell(i,22).value))
                            self.res3.append((DATE, sheet.cell(i,20).value, sheet.cell(i,23).value))
            dt += timedelta(1)

    def writeMatsuba(self):
        df_res2 = pd.DataFrame(self.res2, columns=['DATE','HIGH','LOW'])
        df_res3 = pd.DataFrame(self.res3, columns=['DATE','HIGH','LOW'])
        df_res2.to_csv(self.path+self.file[0],date_format="%Y-%m-%d",index=False)
        df_res3.to_csv(self.path+self.file[1],date_format="%Y-%m-%d",index=False)

    def run(self):
        if not self.prepareDirectory():
            return
        self.getMatsuba()
        self.writeMatsuba()


def writeLog(content):
    logfile = "log_convert_d-matsuba.txt"
    f = open(logfile, 'a')
    f.write(str(datetime.now())+' '+str(content)+'\n')
    f.close()


if __name__ == "__main__":
    writeLog("-----------------------------------------------------------------------")
    threads = []
    for ticker in TICKER:
        task = getMatsubadaily(ticker)
        threads.append(task)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print "Waiting for finish..."
    writeLog("-----------------------------------------------------------------------\n")











