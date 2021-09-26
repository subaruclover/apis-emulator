"""
Create on Sep 26, 2021

@author: Qiong
"""

import logging.config
import time
from datetime import timedelta

logger = logging.getLogger(__name__)

import global_var as gl
import config as conf
import analyser


def lossesAndBatteryFlow(accumulateLosses=False):
    for i in gl.oesunits:

        # ------------ Losses ----------- #

        # ESS loss + DCDC loss
        gl.dcloss[i] = conf.constantSystemLoss + (
                gl.oesunits[i]["dcdc"]["meter"]["wg"] - gl.oesunits[i]["dcdc"]["meter"]["wb"])

        # Trans loss
        gl.acloss[i] = conf.transLoss_c

        # add mode specific UPS losses
        if gl.is_bypassMode[i]:
            gl.acloss[i] += conf.bypassModeLoss_c + (conf.bypassModeLoss_a + conf.transLoss_a) * gl.oesunits[i]["emu"][
                "ups_output_power"]
            gl.oesunits[i]["dcdc"]["powermeter"]["p2"] = int(gl.oesunits[i]["emu"]["ups_output_power"] + gl.acloss[i])
        else:
            gl.dcloss[i] += conf.battModeLoss_c + conf.battModeLoss_a * gl.oesunits[i]["emu"]["ups_output_power"]
            gl.oesunits[i]["dcdc"]["powermeter"]["p2"] = int(gl.acloss[i])

        # calculate p2 (add ac charging power flow )
        if gl.is_ACCharging[i]:
            gl.acloss[i] += conf.ACChargeLoss
            gl.oesunits[i]["dcdc"]["powermeter"]["p2"] += int(conf.ACChargeAmount + conf.ACChargeLoss)

        # ------------ Battery Power Flow including losses----------- #
        outpower = gl.oesunits[i]["emu"]["ups_output_power"] - gl.oesunits[i]["dcdc"]["meter"]["wg"] + gl.dcloss[i]

        # battery is already full but pv produces more than outgoing power
        if gl.oesunits[i]["emu"]["rsoc"] >= 100 and outpower < gl.oesunits[i]["emu"]["pvc_charge_power"]:
            gl.oesunits[i]["emu"]["charge_discharge_power"] = 0
            gl.wasted[i] = gl.oesunits[i]["emu"]["pvc_charge_power"] - outpower
            if conf.debug:
                logger.debug(
                    i + ": battery full-> rsoc=" + str(gl.oesunits[i]["emu"]["rsoc"]) + ", potential pv " + str(
                        gl.oesunits[i]["emu"]["pvc_charge_power"]) + " but really used " + str(outpower))
            gl.oesunits[i]["emu"]["pvc_charge_power"] = round(outpower, 2)

        # if battery is not yet full or outgoing power is smaller than pv power
        else:
            gl.wasted[i] = 0
            # calculate charge_discharge power (always positive!)
            powerflowToBattery = gl.oesunits[i]["dcdc"]["meter"]["wg"] + \
                                 gl.oesunits[i]["emu"]["pvc_charge_power"] + \
                                 gl.oesunits[i]["dcdc"]["powermeter"]["p2"] - \
                                 gl.oesunits[i]["emu"]["ups_output_power"] - \
                                 gl.acloss[i] - \
                                 gl.dcloss[i]
            # batteryFlow is always positive !
            # if powerflowToBattery > 0:
            #     gl.oesunits[i]["emu"]["charge_discharge_power"] = round(powerflowToBattery, 2)
            #     gl.oesunits[i]["emu"]["battery_current"] = round(
            #         gl.oesunits[i]["emu"]["charge_discharge_power"] / conf.batteryVoltage, 2)
            #     # logger.debug( i+ ": charge_disch "+ str(gl.oesunits[i]["emu"]["charge_discharge_power"]) + ", ACLoss: "+str(ACLoss) + ", DCLoss: " +str(DCLoss))
            # else:
            #     gl.oesunits[i]["emu"]["charge_discharge_power"] = - round(powerflowToBattery, 2)
            #     gl.oesunits[i]["emu"]["battery_current"] = -round(
            #         gl.oesunits[i]["emu"]["charge_discharge_power"] / conf.batteryVoltage, 2)
            #     # logger.debug( i+ ": charge_disch "+ str(-gl.oesunits[i]["emu"]["charge_discharge_power"]) + ", ACLoss: "+str(ACLoss) + ", DCLoss: " +str(DCLoss))

            # TODO RL Learner =====
            # first try with fixed battery char/dischar current
            if powerflowToBattery > 0:
                gl.oesunits[i]["emu"]["charge_discharge_power"] = round(powerflowToBattery, 2)
                gl.oesunits[i]["emu"]["battery_current"] = round(
                    3000 / conf.batteryVoltage, 2)
                # logger.debug( i+ ": charge_disch "+ str(gl.oesunits[i]["emu"]["charge_discharge_power"]) + ", ACLoss: "+str(ACLoss) + ", DCLoss: " +str(DCLoss))
            else:
                gl.oesunits[i]["emu"]["charge_discharge_power"] = - round(powerflowToBattery, 2)
                gl.oesunits[i]["emu"]["battery_current"] = -round(
                    1000 / conf.batteryVoltage, 2)
                # logger.debug( i+ ": charge_disch "+ str(-gl.oesunits[i]["emu"]["charge_discharge_power"]) + ", ACLoss: "+str(ACLoss) + ", DCLoss: " +str(DCLoss))


from bottle import run, route, template


@route("/")
def index():
    username = 'test name'
    return template('sample', username=username)


if __name__ == "__main__":
    run(host='localhost', port=8080, reloader=True, debug=True)