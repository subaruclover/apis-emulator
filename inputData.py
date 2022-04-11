"""
Created on Aug 6, 2015

@author: annette
TODO: change our own data instead of sample data
data format stays the same
"""

# import pandas as pd
import numpy as np
import logging.config, datetime
from copy import deepcopy
import global_var as gl
import config as conf

logger = logging.getLogger(__name__)


class inputDataManager():

    def __init__(self, inputSet):
        # demandData = demandData
        # solarData = solarData
        # loadInputdata
        # if inputSet == "Sample":
        if inputSet == "Oist":
            gl.startTime = datetime.datetime(2020, 5, 8, 0, 0, 0)
            gl.endTime = datetime.datetime(2020, 6, 8, 0, 0, 0)
            # gl.startTime = datetime.datetime(2019, 10, 1, 0, 0, 0)
            # gl.endTime = datetime.datetime(2020, 10, 31, 23, 59, 59)
            gl.now = deepcopy(gl.startTime)
            # loadSample
            # old_loadDemand_Sample()  #
            Load_data()
            # define demand update
            # self.demandUpdate = old_demandUpdate_Sample  #
            self.demandUpdate = loadUpdate

            # load solar radiation data
            # loadSol_Sample()  #
            PV_data()

            # define PV update
            # self.pvcUpdate = old_pvcUpdate_Sample  #
            self.pvcUpdate = PVUpdate

            for emulid in gl.displayNames:
                conf.batterySize[emulid] = conf.default_batterySize
                conf.pvc_sol_reg[emulid] = conf.default_Area * conf.r * conf.pr  # seems not effect in sol power update?

        else:
            logger.error("Could not read input data for " + inputSet)


############################
# load data from CSV, sample data
############################

def loadSol_Sample():
    global sol
    # sol_data = pd.read_csv('data/input/Sample/sample_solar_data.csv')
    # sol = pd.np.array(sol_data) 

    # print('#### sol ({}) ####'.format(len(sol)))
    # for a in sol:
    #   print(*a, sep=',')

    sol = np.loadtxt('data/input/Sample/sample_solar_data.csv', delimiter=',')  # [unit: W/m^2]

    # print('#### sol ({}) ####'.format(len(sol)))
    # for a in sol:
    #   print(*a, sep=',')

    return sol


def old_loadDemand_Sample():
    global demand
    # demand = {}
    # demand_data = pd.read_csv('data/input/Sample/sample_load_data.csv')

    # cusids = set(demand_data.ix[:,1])
    # for i, cusid in enumerate(cusids):
    #     # takes all values per userid with a step of 2 [colum 6 to 53]
    #     demand_cusid = demand_data.ix[demand_data.ix[:,1]==cusid, range(6,len(demand_data.ix[0,:]),2)]
    #     cus_id = "E{0:03d}".format(i+1)
    #     demand[cus_id] = pd.np.array(demand_cusid)
    #     gl.displayNames[cus_id]="Sample_"+cus_id

    # print('#### cusids : {}'.format(cusids))
    # for key in demand.keys():
    #     a = demand[key]
    #     print('#### demand[{}] ({})'.format(key, len(a)))
    #     for aa in a:
    #         print(*aa, sep=',')

    demand = {}
    # demand_data = np.genfromtxt('data/input/Sample/sample_load_data.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    cols = list(range(6, 52 + 1, 2))
    cols.insert(0, 1)
    # read column 2, col 7~53 for every 2 cols (1 hour per data point) from input data
    # replace our own data files to the /Sample/ folder
    demand_data = np.loadtxt('data/input/Sample/sample_load_data.csv', delimiter=',', skiprows=1, usecols=cols)  # [unit: kW]
    for row in demand_data:
        cus_id = "E{0:03d}".format(int(row[0]))
        if not demand.get(cus_id):
            demand[cus_id] = []
        demand[cus_id].append(row[1:])
    for cus_id in demand:
        demand[cus_id] = np.array(demand[cus_id])
        gl.displayNames[cus_id] = "Sample_" + cus_id

    # print('#### cols : {}'.format(cols))
    # print('#### demand_data : {}'.format(demand_data))
    # for key in demand.keys():
    #     a = demand[key]
    #     print('#### demand[{}] ({})'.format(key, len(a)))
    #     for aa in a:
    #         print(*aa, sep=',')

    return demand


############################
# load data from CSV with our data
############################

def PV_data():  # load house's PV production data
    global pv
    pv = {}

    cols = list(range(2, 2880 + 2, 1))
    cols.insert(0, 0)
    # read column 0, col 2~2882(end) for each cols (30s per data point) from input data
    pv_data = np.loadtxt('data/input/Oist/fourhouses_2019_apis_sol_reform_May.csv', delimiter=',', skiprows=1, usecols=cols)

    for row in pv_data:
        # print(int(row[0]), row)
        cus_id = "E{0:03d}".format(int(row[0]))
        # print("cus_id", cus_id)
        # print(type(pv))
        if not pv.get(cus_id):
            pv[cus_id] = []
        pv[cus_id].append(row[1:])
        # print(pv[cus_id])

    #     print("all id", demand)

    for cus_id in pv:
        pv[cus_id] = np.array(pv[cus_id])
        gl.displayNames[cus_id] = "Oist_" + cus_id

    return pv


def Load_data():  # load oist house's comsumption data
    global consumption
    consumption = {}

    cols = list(range(2, 2880 + 2, 1))
    cols.insert(0, 0)
    # read column 0, col 2~2881(end) for each cols (30s per data point) from input data
    consumption_data = np.loadtxt('data/input/Oist/fourhouses_2019_apis_load_reform_May.csv', delimiter=',',
        skiprows=1, usecols=cols)

    for row in consumption_data:
        cus_id = "E{0:03d}".format(int(row[0]))
        if not consumption.get(cus_id):
            consumption[cus_id] = []
        consumption[cus_id].append(row[1:])

    for cus_id in consumption:
        consumption[cus_id] = np.array(consumption[cus_id])
        gl.displayNames[cus_id] = "Oist_" + cus_id

    return consumption


######################
# update functions to be used by emulator with sample data
######################

def old_pvcUpdate_Sample():
    count_h = float(gl.count_s) / 3600
    weight = count_h - int(count_h)
    step_now = (int((count_h) / 24)), int((count_h) % 24)
    step_next = (int((count_h + 1) / 24)), int((count_h + 1) % 24)
    if int(count_h + 1) >= sol.size:
        logger.debug("no more solar radiation data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["pvc_charge_power"] = round((1 - weight) * sol[step_now] + weight * sol[step_next],
                                                              2)  # sol[W]
    return True


def old_demandUpdate_Sample():
    count_h = float(gl.count_s) / 3600
    weight = count_h - int(count_h)
    step_now = int((count_h) / 24), int((count_h) % 24)
    step_next = (int((count_h + 1) / 24), int((count_h + 1) % 24))
    if int(count_h + 1) >= demand[next(iter(gl.oesunits))].size:
        logger.debug("no more demand data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["ups_output_power"] = \
            round(((1 - weight) * demand[oesid][step_now] + weight * demand[oesid][step_next]) * 1000, 2)  # demand[W]
    return True


######################
# update functions to be used by emulator with our data
# our data is 30s for each house
######################
"""
def PVCUpdate():
    # count_s = 3600*12 # how many seconds have passed
    count_t = float(count_s) / 30  # set counter for data which is collected every 30s
    weight = count_t - int(count_t)
    step_now = (int((count_t) / 2880) + 1) * int((count_t) % 2880)
    step_next = (int((count_t + 1) / 2880) + 1) * int((count_t + 1) % 2880)

    if int(count_t + 1) >= our_data.size:
        logger.debug("no more our_data radiation data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["pvc_charge_power"] = round(
            (1 - weight) * our_data[step_now] + weight * our_data[step_next], 2)  # sol[W]

    # print("our_data[step_now]", our_data[step_now], "\n", "our_data[step_next]", our_data[step_next])
    # print("pvc power", round((1 - weight) * our_data[step_now] + weight * our_data[step_next], 2))
    # print((1 - weight) * our_data[step_now] + weight * our_data[step_next])

    return True
"""


def PVUpdate():
    count_t = float(gl.count_s) / 30  # set counter for data which is collected every 30s
    weight = count_t - int(count_t)
    step_now = int((count_t) / 2880), int((count_t) % 2880)
    step_next = (int((count_t + 1) / 2880), int((count_t + 1) % 2880))

    # step_now = (int((count_t) / 2880) ) * int((count_t) % 2880)
    # step_next = (int((count_t + 1) / 2880) ) * int((count_t + 1) % 2880)

    if int(count_t + 1) >= pv[next(iter(gl.oesunits))].size:
        logger.debug("no more oist radiation data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["pvc_charge_power"] = round(
            (1 - weight) * pv[oesid][step_now] + weight * pv[oesid][step_next], 2)  # our_data[W]
        with open("data/output/output.txt", "a") as f:
            print(oesid, "time: ", gl.now, "sol data update: ", gl.oesunits[oesid]["emu"]["pvc_charge_power"], file=f)

    # print("our_data[step_now]", our_data[step_now], "\n", "our_data[step_next]", our_data[step_next])
    # print("pvc power", round((1 - weight) * our_data[step_now] + weight * our_data[step_next], 2))
    # print((1 - weight) * our_data[step_now] + weight * our_data[step_next])

    return True


def loadUpdate():
    count_t = float(gl.count_s) / 30  # set counter for data which is collected every 30s
    weight = count_t - int(count_t)
    step_now = int((count_t) / 2880), int((count_t) % 2880)
    step_next = (int((count_t + 1) / 2880), int((count_t + 1) % 2880))

    # step_now = (int((count_t) / 2880) ) * int((count_t) % 2880)
    # step_next = (int((count_t + 1) / 2880) ) * int((count_t + 1) % 2880)

    if int(count_t + 1) >= consumption[next(iter(gl.oesunits))].size:
        logger.debug("no more oist consumption data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["ups_output_power"] = round(
            (1 - weight) * consumption[oesid][step_now] + weight * consumption[oesid][step_next], 2)  # consumption_data[W]

    ########
    # count_h = float(gl.count_s) / 3600
    # weight = count_h - int(count_h)
    # step_now = int((count_h) / 24*4), int((count_h) % 24*4)
    # step_next = (int((count_h + 1) / 24*4), int((count_h + 1) % 24*4))
    # if int(count_h + 1) >= consumption[next(iter(gl.oesunits))].size:
    #     logger.debug("no more consumption data")
    #     return False
    # for oesid in gl.oesunits:
    #     gl.oesunits[oesid]["emu"]["ups_output_power"] = round(
    #         ((1 - weight) * consumption[oesid][step_now] + weight * demand[oesid][step_next]), 2)  # consumption[W]

    return True
