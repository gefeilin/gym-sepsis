import gym
import numpy as np
import os
import pandas as pd
from collections import deque
from tensorflow import keras
from gym import spaces

STATE_MODEL = "model/sepsis_states.h5"
TERMINATION_MODEL = "model/sepsis_termination.h5"
OUTCOME_MODEL = "model/sepsis_outcome.h5"
STARTING_STATES_VALUES = "model/sepsis_starting_states.npz"

NUM_FEATURES = 48  # 46 + action + state index
NUM_ACTIONS = 24
EPISODE_MEMORY = 10

features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
            'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
            'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
            'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
            'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
            'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
            'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
            'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
            'blood_culture_positive', 'action', 'state_idx']


class SepsisEnv(gym.Env):
    """ Sepsis simulation environment built on MIMIC-trained models. """
    metadata = {'render_modes': ['ansi']}

    def __init__(self, starting_state=None, verbose=False):
        super().__init__()
        module_path = os.path.dirname(__file__)
        self.verbose = verbose

        self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL), compile=False)
        self.termination_model = keras.models.load_model(os.path.join(module_path, TERMINATION_MODEL), compile=False)
        self.outcome_model = keras.models.load_model(os.path.join(module_path, OUTCOME_MODEL), compile=False)

        self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(NUM_FEATURES - 2,), dtype=np.float32
        )

        self.reset(starting_state=starting_state)

    def step(self, action):

        self.memory.append(np.append(np.append(self.s.flatten(), action), self.state_idx))
        memory_array = np.expand_dims(self.memory, 0).astype(np.float32)

        next_state = self.state_model.predict(memory_array[:, :, :-1], verbose=0)

        constants = ['age', 'race_white', 'race_black', 'race_hispanic', 'race_other', 'height', 'weight']
        for c in constants:
            idx = features.index(c)
            next_state[0, idx] = self.state_0[idx]

        
        termination = self.termination_model.predict(memory_array, verbose=0)
        outcome = self.outcome_model.predict(memory_array, verbose=0)

        termination_state = np.argmax(termination)
        outcome_state = np.argmax(outcome)

        reward, terminated, truncated = 0, False, False
        if termination_state == 1:  # "done"
            terminated = True
            reward = -15 if outcome_state == 0 else 15

        
        self.s = next_state[0].astype(np.float32)[:NUM_FEATURES - 2]
        self.state_idx += 1
        self.rewards.append(reward)
        self.dones.append(terminated)

        return self.s, reward, terminated, truncated, {"prob": 1.0}

    def reset(self, starting_state=None, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rewards, self.dones = [], []
        self.state_idx = 0
        self.memory = deque([np.zeros(NUM_FEATURES)] * EPISODE_MEMORY, maxlen=EPISODE_MEMORY)

        if starting_state is None:
            self.s = self.starting_states[np.random.randint(len(self.starting_states))][:-1]
        else:
            self.s = starting_state

        self.s = self.s.astype(np.float32).flatten()
        self.state_0 = np.copy(self.s)

        if self.verbose:
            print("starting state:", self.s)

        return self.s, {}

    def render(self, mode='ansi'):
        df = pd.DataFrame(list(self.memory), columns=features)
        print(df)