'''
Created on Aug 6, 2015

@author: annette
'''

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
        if inputSet == "Sample":
            gl.startTime = datetime.datetime(2020, 1, 1, 0, 0, 0)
            gl.endTime = datetime.datetime(2021, 1, 1, 0, 0, 0)
            # gl.startTime = datetime.datetime(2019, 1, 1, 0, 0, 0)
            # gl.endTime = datetime.datetime(2019, 12, 31, 0, 0, 0)
            gl.now = deepcopy(gl.startTime)
            # loadSample
            old_loadDemand_Sample()
            # Load_data()
            # define demand update
            self.demandUpdate = old_demandUpdate_Sample
            # self.demandUpdate = loadUpdate()

            # load solar radiation data
            loadSol_Sample()
            # PV_data()
            # define PV update
            self.pvcUpdate = old_pvcUpdate_Sample
            # self.pvcUpdate = pvUpdate()

            for emulid in gl.displayNames:
                # TODO: replace with pvc_charge_power for our data
                conf.batterySize[emulid] = conf.default_batterySize
                conf.pvc_sol_reg[emulid] = conf.default_Area * conf.r * conf.pr

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

    sol = np.loadtxt('data/input/Sample/sample_solar_data.csv', delimiter=',')

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
    cols = list(range(6, 52+1, 2))
    cols.insert(0, 1)
    # read column 2, col 7~53 for every 2 cols (1 hour per data point) from input data
    # replace our own data files to the /Sample/ folder
    demand_data = np.loadtxt('data/input/Sample/sample_load_data.csv', delimiter=',', skiprows=1, usecols=cols)
    for row in demand_data:
        cus_id = "E{0:03d}".format(int(row[0]))
        if not demand.get(cus_id):
            demand[cus_id] = []
        demand[cus_id].append(row[1:])
    for cus_id in demand:
        demand[cus_id] = np.array(demand[cus_id])
        gl.displayNames[cus_id] = "Sample_"+cus_id

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

    col = list(range(2, 6, 1))
    col.insert(0, 0)
    # print(col)
    our_data = np.loadtxt('data/input/Sample/house214_2019.csv', delimiter=',', skiprows=1, usecols=col)
    # get the pvc_charge_power column (3rd), house 214, 2019
    # pv = our_data[:, 2]

    pv = {}

    day_len = int(len(our_data) / 96)

    for load_col in our_data:
        house_id = "E{0:03d}".format(int(load_col[0]))
        if not pv.get(house_id):
            pv[house_id] = []

    for day in range(day_len):
        pv[house_id].append(our_data[day * 96:(day + 1) * 96, 2])

    for house_id in pv:
        pv[house_id] = np.array(pv[house_id])

    pv = pv[house_id]

    return pv


def Load_data():  # load house's consumption data
    global consumption

    consumption = {}
    col = list(range(2, 6, 1))
    col.insert(0, 0)
    # print(col)
    our_data = np.loadtxt('data/input/Sample/house214_2019.csv', delimiter=',', skiprows=1, usecols=col)
    # get the load column (2nd), house 214, 2019
    # consumption = our_data[:, 1]

    day_len = int(len(our_data) / 96)

    for load_col in our_data:
        house_id = "E{0:03d}".format(int(load_col[0]))
        if not consumption.get(house_id):
            consumption[house_id] = []

    for day in range(day_len):
        consumption[house_id].append(our_data[day * 96:(day + 1) * 96, 1])

    for house_id in consumption:
        consumption[house_id] = np.array(consumption[house_id])
        gl.displayNames[house_id]="Sample_" + house_id

    return consumption


######################
# update functions to be used by emulator with sample data
######################

def old_pvcUpdate_Sample():
    count_h = float(gl.count_s)/3600
    weight = count_h-int(count_h)
    step_now = (int((count_h) / 24)), int((count_h) % 24)
    step_next = (int((count_h+1)/24)), int((count_h+1) % 24)
    if int(count_h+1) >= sol.size:
        logger.debug("no more solar radiation data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["pvc_charge_power"] = round((1-weight)*sol[step_now] + weight*sol[step_next], 2)  # sol[W]
    return True

def old_demandUpdate_Sample():
    count_h = float(gl.count_s)/3600
    weight = count_h-int(count_h)
    step_now = int((count_h) / 24), int((count_h) % 24)
    step_next = (int((count_h+1) / 24), int((count_h+1) % 24))
    if int(count_h+1) >= demand[next(iter(gl.oesunits))].size:
        logger.debug("no more demand data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["ups_output_power"] = \
            round(((1-weight)*demand[oesid][step_now] + weight*demand[oesid][step_next])*1000, 2)  # demand[W]
    return True


######################
# update functions to be used by emulator with our data
######################
def pvUpdate():
    count_h = float(gl.count_s) / 3600
    weight = count_h - int(count_h)
    step_now = (int((count_h) / 24*4)), int((count_h) % 24*4)
    step_next = (int((count_h + 1) / 24*4)), int((count_h + 1) % 24*4)
    if int(count_h + 1) >= pv.size:
        logger.debug("no more pv production data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["pvc_charge_power"] = round((1 - weight) * pv[step_now] + weight * pv[step_next],
                                                              2)  # sol[W]
    return True


def loadUpdate():
    count_h = float(gl.count_s) / 3600
    weight = count_h - int(count_h)
    step_now = int((count_h) / 24*4), int((count_h) % 24*4)
    step_next = (int((count_h + 1) / 24*4), int((count_h + 1) % 24*4))
    if int(count_h + 1) >= consumption[next(iter(gl.oesunits))].size:
        logger.debug("no more consumption data")
        return False
    for oesid in gl.oesunits:
        gl.oesunits[oesid]["emu"]["ups_output_power"] = round(
            ((1 - weight) * consumption[oesid][step_now] + weight * demand[oesid][step_next]), 2)  # consumption[W]
    return True