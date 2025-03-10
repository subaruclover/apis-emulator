"""
Created on Aug 3, 2015

@author: annette
@modified: Qiong
"""

####################
# Running config
####################

debug = False

# input data
# dataSet = "Sample"
dataSet = "Oist"

# sleep time between execution loops
# Pseudo offline mode, for speeding up
# sleeptime = 1  =>  1s of emulator = gl.acc seconds of real time
# sleeptime = 0  =>  run at processor speed
sleeptime = 1

# start emulator for emulating powerflow of whole physical system (PV, battery, ...)
doUpdates = True
# use individualConfig.csv file for configuring battery and PV size for each house
individualConf = True

# save indivudual status to DB (similar table as in reality)
saveIndividualToDB = False
# save cummulative summary to csv file
saveToSummaryToCSV = True
# save all the individual houses status to CSV file
saveIndividualToCSV = True


# # #################
# # # losses
# # #################
# 
# #loss = a*load +c
constantSystemLoss = 17.5
# 
       
# DCDC converter losses
DCCstLoss = 4.5
DCChargeLoss = 71
DCDischargeLoss = 55
#     
# UPS battery_mode loss taken from DC
battModeLoss_a = 0.140 
battModeLoss_c = 129
              
# UPS battery_mode loss taken from AC 
transLoss_a = 0.027
transLoss_c = 57
              
# bypass and trans loss taken from AC
bypassModeLoss_a = 0.033
bypassModeLoss_c = 67
             
# AC Charge loss taken from AC
ACChargeLoss = 10  # ???
         
# ========== UPS SETUP ========#
UPS_TRIGGER_BATT_OFF = 20  # 30
UPS_TRIGGER_BATT_ON = 25  # 35
UPS_TRIGGER_AC_OFF = 15  # 25
UPS_TRIGGER_AC_ON = 10  # 20


####################
# System setup config
####################
# voltage
batteryVoltage = 48

# ac charge limit
ACChargeAmount = 150  # 350*2.5

# default battery size
default_batterySize = 4800.0

# PV size
default_Area = 18 * 1.2  # default Area
r = 0.19  # panel yield
pr = 0.9  # performance factor

# pv and battery size for each house (initialized in inputData)
batterySize = {}
pvc_sol_reg = {}

#######################
# runtime config
#######################

dburl = 'http://temp.com'
# one year
# summaryPath = 'data/output/summary_sample.csv'
# indivLogPath = 'data/output/indivLog_sample.csv'
# 2 months
# summaryPath = 'data/output/summary_sample_action.csv'
# indivLogPath = 'data/output/indivLog_sample_action.csv'

# one month
# summaryPath = 'data/output/oist_summary_May_default_3.csv'
# indivLogPath = 'data/output/oist_indivLog_May_default_3.csv'
# summaryPath = 'data/output/oist_summary_May_iter1_shuffle.csv'
# indivLogPath = 'data/output/oist_indivLog_May_iter1_shuffle.csv'
summaryPath = 'data/output/oist_summary_May_Prior_iter3_run3_time.csv'
indivLogPath = 'data/output/oist_indivLog_May_Prior_iter3_run3_time.csv'

b_host = "0.0.0.0"
b_port = 4390
# set_url = ":4380/remote/set?"

# Modes
modes = {
"0x0000": 'Waiting',
"0x0002": 'Heteronomy CV Discharging', 
"0x0014": 'Grid Autonomy Mode',
"0x0041": 'Heteronomy CV Charging - grid current limited'
}
modesOps = {
"0x0000": 'Waiting',
"0x0002": 'Heteronomy CV', 
"0x0014": 'Grid Autonomy',
"0x0041": 'Heteronomy CV'
}
modesRunning = {
"0x0000": 'off',
"0x0002": 'discharge', 
"0x0014": 'discharge',  # grid autonomy mode can charge and discharge but we only use it in discharging mode
"0x0041": 'charge'
}
