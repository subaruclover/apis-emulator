"""
DQN training, single run

created by: Qiong
09/28/2021
"""
import time

import pickle
import numpy as np
import pandas as pd

import global_var as gl
import config as conf


import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from RL_learn import DQNNet, SumTree, Memory, BatteryEnv

start_time = time.time()

##############################
# Data loading
host = conf.b_host
port = conf.b_port
# url = "http://0.0.0.0:4390/get/log"

URL = "http://" + host + ":" + str(port) + "/get/log"

import requests, json
# response = requests.request("POST", url, data=gl)
# print(response.text)
# dicts of states for all houses
pvc_charge_power = {}
ups_output_power = {}
p2 = {}  # powermeter.p2, Power consumption to the power storage system [W]
wg = {}  # meter.wg, DC Grid power [W]
wb = {}  # meter.wb, Battery Power [W]

# need to refresh the output data every 5s? time.sleep()
while not gl.sema:  # True
    # # refresh every 5 seconds
    # time.sleep(5)
    # read variables from /get/log url
    # print(output_data.text)
    output_data = requests.get(URL).text
    output_data = json.loads(output_data)  # dict

    for ids, dict_ in output_data.items():  # ids: E001, E002, ... house ID
        # print('the name of the dictionary is ', ids)
        # print('the dictionary is ', dict_)
        pvc_charge_power[ids] = output_data[ids]["emu"]["pvc_charge_power"]
        ups_output_power[ids] = output_data[ids]["emu"]["ups_output_power"]
        p2[ids] = output_data[ids]["dcdc"]["powermeter"]["p2"]
        wg[ids] = output_data[ids]["dcdc"]["meter"]["wg"]
        wb[ids] = output_data[ids]["dcdc"]["meter"]["wb"]

        print("pv of {ids} is {pv},".format(ids=ids, pv=pvc_charge_power[ids]),
              "load of {ids} is {load},".format(ids=ids, load=ups_output_power[ids]),
              "p2 of {ids} is {p2},".format(ids=ids, p2=p2[ids]),
              "wg of {ids} is {wg},".format(ids=ids, wg=wg[ids]),
              "wb of {ids} is {wb},".format(ids=ids, wb=wb[ids])
              )


    # refresh every 5 seconds
    time.sleep(5)

    # State concatenate
    pv = np.array(pvc_charge_power)
    load = np.array(ups_output_power)
    p2 = np.array(p2)

    x = np.concatenate([pv, load, p2], axis=-1)




# input data of house 214, 2019 (every 15mins (quarter hour), each day (pu, per unit) contains 96 data points)
# df_raw = pd.read_csv("/home/doya/Documents/DQNBattery/data/house214_2019_quarterhour_avg.csv")
# # df_raw = pd.read_csv("./data/house214_2019_quarterhour_avg.csv")
# df = df_raw.fillna(method="ffill", inplace=False)  # replace NaN data with forward vaild data
# # T = 96 * 50
# df = df[:T]

charge_discharge_power = df["charge_discharge_power"]  # W
rsoc = df["rsoc"]  # %
pvc_charge_power = df["pvc_charge_power"]  # W
battery_current = df["battery_current"]  # A(DC)
p2 = df["p2"]  # W
ups_output_power = df["ups_output_power"]


#############################
# State concatenate
pv = df[["pvc_charge_power"]].values
load = df[["ups_output_power"]].values
p2 = df[["p2"]].values

x = np.concatenate([pv, load, p2], axis=-1)

# Parameters
# DQN hyperparameters
state_size = (4,)  # rsoc, pv_power, consumption
action_size = 15
learning_rate = 0.01

# Training hyperparameters
batch_size = 256
EPI = 10

# Exploration hyperparameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.001  # exponential decay rate for exploration prob

# Q-learning hyperparameters
gamma = 0.96  # Discounting rate of future reward

# Memory hyperparameters
pretrain_length = 10000  # # of experiences stored in Memory during initialization
memory_size = 10000  # # of experiences Memory can keep

battery = BatteryEnv(action_size=action_size)

memory = Memory(memory_size)

np.random.seed(42)

##################################
# Memory initialization
RSOC = np.array([battery.initial_rsoc])
day = 0
quarter_hour = 0
done = False
timestep = 15.0

for i in range(pretrain_length):

    state = np.concatenate((x[day * 96 + quarter_hour, :], RSOC), axis=-1)
    action = np.random.randint(0, action_size)

    # Compute the reward and new state based on the selected action
    # next_rsoc, reward
    next_rsoc, reward, p2_sim, prod = battery.step(state, action, timestep)
    #     print('next_rsoc: ', next_rsoc, 'reward: ', reward)

    # Store the experience in memory
    if quarter_hour < 96 - 1:
        quarter_hour += 1
        next_state = np.concatenate((x[day * 96 + quarter_hour, :], next_rsoc), axis=-1)
    else:
        done = True
        day += 1
        quarter_hour = 0
        if day < len(x) / 96:
            next_state = np.concatenate(
                (x[day * 96 + quarter_hour, :], next_rsoc), axis=-1
            )
        else:
            break

    RSOC = next_rsoc
    experience = state, action, reward, next_state, done
    memory.store(experience)


#########################################
# DQN Training

DQN = DQNNet(
    state_size=state_size, action_size=action_size, learning_rate=learning_rate
)

decay_step = 0  # Decay rate for ϵ-greedy policy
RSOC = np.array([battery.initial_rsoc])
day = 0
quarter_hour = 0
done = False
timestep = 15.0
quarter_hour_rewards = []
day_mean_rewards = []

while day < len(x) / 96:

    state = np.concatenate((x[day * 96 + quarter_hour, :], RSOC), axis=-1)

    # ϵ-greedy policy
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(
        -decay_rate * decay_step
    )
    if explore_probability > exp_exp_tradeoff:
        action = np.random.randint(0, action_size)
    else:
        action = np.argmax(DQN.model.predict(np.expand_dims(state, axis=0)))

    # Compute the reward and new state based on the selected action
    next_RSOC, reward, p2_sim, prod = battery.step(state, action, timestep)
    #   print('next_rsoc: ', next_RSOC, 'reward: ', reward)

    quarter_hour_rewards.append(reward)

    # Store the experience in memory
    if quarter_hour < 96 - 1:
        quarter_hour += 1
        next_state = np.concatenate((x[day * 96 + quarter_hour, :], next_RSOC), axis=-1)
    else:
        done = True
        day += 1
        quarter_hour = 0
        if day < len(x) / 96:
            next_state = np.concatenate(
                (x[day * 96 + quarter_hour, :], next_RSOC), axis=-1
            )
        else:
            break
        mean_reward = np.mean(quarter_hour_rewards)
        day_mean_rewards.append(mean_reward)
        quarter_hour_rewards = []
        print(
            "Day: {}".format(day),
            "Mean reward: {:.2f}".format(mean_reward),
            "Training loss: {:.2f}".format(loss),
            "Explore P: {:.2f} \n".format(explore_probability),
        )

    RSOC = next_RSOC
    experience = state, action, reward, next_state, done
    memory.store(experience)
    decay_step += 1

    # DQN training
    tree_idx, batch, ISWeights_mb = memory.sample(
        batch_size
    )  # Obtain random mini-batch from memory

    states_mb = np.array([each[0][0] for each in batch])
    actions_mb = np.array([each[0][1] for each in batch])
    rewards_mb = np.array([each[0][2] for each in batch])
    next_states_mb = np.array([each[0][3] for each in batch])
    dones_mb = np.array([each[0][4] for each in batch])

    targets_mb = DQN.model.predict(states_mb)

    #     print('s_mb:',states_mb, 'a_mb:', actions_mb, 'r_mb:', rewards_mb)

    # Update those targets at which actions are taken
    target_batch = []
    q_next_state = DQN.model.predict(next_states_mb)
    for i in range(0, len(batch)):
        action = np.argmax(q_next_state[i])
        if dones_mb[i] == 1:
            target_batch.append(rewards_mb[i])
        else:
            target = rewards_mb[i] + gamma * q_next_state[i][action]
            target_batch.append(rewards_mb[i])

    # Replace the original with the updated targets
    one_hot = np.zeros((len(batch), action_size))
    one_hot[np.arange(len(batch)), actions_mb] = 1
    targets_mb = targets_mb.astype("float64")
    target_batch = np.array([each for each in target_batch]).astype("float64")
    np.place(targets_mb, one_hot > 0, target_batch)

    loss = DQN.model.train_on_batch(
        states_mb, targets_mb, sample_weight=ISWeights_mb.ravel()
    )

    # Update priority
    absolute_errors = []
    predicts_mb = DQN.model.predict(states_mb)
    for i in range(0, len(batch)):
        absolute_errors.append(
            np.abs(predicts_mb[i][actions_mb[i]] - targets_mb[i][actions_mb[i]])
        )
    absolute_errors = np.array(absolute_errors)

    tree_idx = np.array([int(each) for each in tree_idx])
    memory.batch_update(tree_idx, absolute_errors)

    # Save model every 5 days
    if day % 5 == 0:
        # DQN.model.save_weights("/Users/Huang/Documents/DQNBattery/DQN_quarterhour_avg_214.hdf5")
        DQN.model.save_weights("./DQN_quarterhour_avg_214.hdf5")


############################################
# Testing
# DQN.model.load_weights("/Users/Huang/Documents/DQNBattery/DQN_quarterhour_avg_214.hdf5")
DQN.model.load_weights("./DQN_quarterhour_avg_214.hdf5")


RSOC = np.array([battery.initial_rsoc])
day = 0
quarter_hour = 0
done = False
timestep = 15.0
RSOC_list = []
action_list = []
reward_list = []
p2_sim_list = []
prod_list = []

while day < len(x) / 96:

    state = np.concatenate((x[day * 96 + quarter_hour, :], RSOC), axis=-1)
    action = np.argmax(DQN.model.predict(np.expand_dims(state, axis=0)))

    next_RSOC, reward, p2_sim, prod = battery.step(state, action, timestep)
    #     print('next_rsoc: ', next_RSOC, 'reward: ', reward)

    RSOC = next_RSOC
    RSOC_list.append(RSOC)
    #     RSOC_list.append(RSOC/100)
    reward_list.append(reward)
    action_list.append(action)
    p2_sim_list.append(p2_sim)
    prod_list.append(prod)

    if quarter_hour < 96 - 1:
        quarter_hour += 1
        next_state = np.concatenate((x[day * 96 + quarter_hour, :], next_RSOC), axis=-1)
    else:
        done = True
        day += 1
        quarter_hour = 0
        if day < len(x) / 96:
            next_state = np.concatenate(
                (x[day * 96 + quarter_hour, :], next_RSOC), axis=-1
            )
        else:
            break

# print(np.mean(reward_list))
end_time = time.time()
print("training time: {:.2f}mins".format((end_time - start_time) / 60))

#################
# plot reward
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

ax.plot(day_mean_rewards, "b-", label="reward")
# ax.plot(mean_reward)

ax.set_xlabel("days of Year 2019 (houe214)", fontsize=14)
ax.set_ylabel("Average reward", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(loc="lower right", fontsize=14)

plt.show()
