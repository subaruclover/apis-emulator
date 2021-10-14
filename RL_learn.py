"""
# Deep Q Network
# Battery Model

created by: Qiong
09/28/2021
"""

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam

# Deep Q-Network Model
class DQNNet():
    def __init__(self, state_size, action_size, learning_rate):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.create_model()

    def create_model(self):

        # state_size = (3, )
        input = Input(shape=self.state_size)

        x = Dense(50, activation="relu", kernel_initializer=glorot_uniform(seed=42))(
            input
        )
        x = Dense(200, activation="relu", kernel_initializer=glorot_uniform(seed=42))(x)

        output = Dense(
            self.action_size,
            activation="linear",
            kernel_initializer=glorot_uniform(seed=42),
        )(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model


# Memory Model
# A tree based array containing priority of each experience for fast sampling
class SumTree():
    """
    __init__ - create data array storing experience and a tree based array storing priority
    add - store new experience in data array and update tree with new priority
    update - update tree and propagate the change through the tree
    get_leaf - find the final nodes with a given priority value

    store data with its priority in the tree.
    """

    data_pointer = 0

    def __init__(self, capacity):

        """
        capacity - Number of final nodes containing experience, for all priority values
        data - array containing experience (with pointers to Python objects), for all transitions
        tree - a tree shape array containing priority of each experience

        tree index:
            0       -> storing priority sum
           / \
          1   2
         / \ / \
        3  4 5  6   -> storing priority for transitions

        Array type for storing:
        [0, 1, 2, 3, 4, 5, 6]
        """

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):

        # Start from first leaf node of the most bottom layer
        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data  # Update data frame
        self.update(tree_index, priority)  # Update priority

        # Overwrite if exceed memory capacity
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):

        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagate the change through tree
        while (
                tree_index != 0
        ):  # this method is faster than the recursive loop in the reference code
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):

        parent_index = 0

        while True:  # while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1  # this leaf's left and right kids
            right_child_index = left_child_index + 1
            # Downward search, always search for a higher priority node till the last layer
            if left_child_index >= len(self.tree):  # reach the bottom, end search
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        # tree leaf index, priority, experience
        return leaf_index, self.tree[leaf_index], self.data[data_index]


# Memory Model
class Memory():  # stored as (s, a, r, s_) in SumTree

    """

    __init__ - create SumTree memory
    store - assign priority to new experience and store with SumTree.add & SumTree.update
    sample - uniformly sample from the range between 0 and total priority and
             retrieve the leaf index, priority and experience with SumTree.get_leaf
    batch_update - update the priority of experience after training with SumTree.update

    PER_e - Hyperparameter that avoid experiences having 0 probability of being taken
    PER_a - Hyperparameter that allows tradeoff between taking only experience with
            high priority and sampling randomly (0 - pure uniform randomness, 1 -
            select experiences with the highest priority)
    PER_b - Importance-sampling, from initial value increasing to 1, control how much
            IS affect learning

    """

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment_per_sampling = 0.01
    absolute_error_upper = 1.0  # Clipped abs error

    def __init__(self, capacity):

        self.tree = SumTree(capacity)

    def store(self, experience):

        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])

        # If the max priority = 0, this experience will never have a chance to be selected
        # So a minimum priority is assigned
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):

        """
        First, to sample a minibatch of k size, the range [0, priority_total] is
        divided into k ranges. A value is uniformly sampled from each range. Search
        in the sumtree, the experience where priority score correspond to sample
        values are retrieved from. Calculate IS weights for each minibatch element
        """

        b_memory = []
        b_idx = np.empty((n,))
        b_ISWeights = np.empty((n, 1))

        priority_segment = self.tree.tree[0] / n

        self.PER_b = np.min([1.0, self.PER_b + self.PER_b_increment_per_sampling])

        prob_min = np.min(self.tree.tree[-self.tree.capacity :]) / self.tree.tree[0]
        max_weight = (prob_min * n) ** (-self.PER_b)

        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            prob = priority / self.tree.tree[0]
            b_ISWeights[i, 0] = (prob * n) ** (-self.PER_b) / max_weight
            b_idx[i] = index
            b_memory.append([data])

        return b_idx, b_memory, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):

        # To avoid 0 probability
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# Battery Model, step function (reward)
class BatteryEnv():

    def __init__(self, action_size):
        """
        coeff_d - discharge coefficient
        coeff_c - charge coefficient

        actions space is 3, where
        a = -1, battery discharge
        a = 0,  battery in idle
        a = 1,  battery charge
        """

        self.action_set = np.linspace(-35, 35, num=action_size, endpoint=True)
        self.initial_rsoc = 30.
        self.battery_voltage = 52.
        self.coeff_c = 0.02
        self.coeff_d = 0.02
        self.decay = 0.001

    def step(self, state, action, timestep):
        current_pv = state[0]
        current_load = state[1]
        current_p2 = state[2]
        current_rsoc = state[3]

        # RSOC -- Bat_cur -> w0, w1
        if self.action_set[action] < 0:  # == -1:   # discharge
            next_rsoc = current_rsoc + (self.coeff_d * self.action_set[action] - self.decay) * timestep
            next_rsoc = np.maximum(next_rsoc, 20.)

        elif self.action_set[action] > 0:  # == 1:   # charge
            next_rsoc = current_rsoc + (self.coeff_c * self.action_set[action] - self.decay) * timestep
            next_rsoc = np.minimum(next_rsoc, 100.)

        else:  # idle
            next_rsoc = current_rsoc - self.decay * timestep
            next_rsoc = np.maximum(next_rsoc, 20.)

        next_rsoc = np.array([next_rsoc])

        battery_charge_power = self.battery_voltage * self.action_set[action]  # battery_output
        p2_sim = current_pv - battery_charge_power
        cost = -p2_sim

        # reward function
        reward = np.minimum(-cost, 0.)
        # reward = -cost

        return next_rsoc, reward, p2_sim, battery_charge_power

