#from DQN_NNb import DQN_NNb
import os
import numpy as np
import pandas as pd
from .Actions import RedispatchActions
from l2rpn_baselines.DoubleDuelingDQN import DoubleDuelingDQN
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class D3QN(DoubleDuelingDQN):

    def __init__(self, observation_space, action_space, name=__name__, is_training=False, num_frames=4, batch_size=32, lr=1e-5):
        super().__init__(observation_space,
                        action_space,
                        name=name,
                        is_training=is_training)
        
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.lr = lr

        red_acts = RedispatchActions(observation_space, action_space)
        self.redispatch_actions_dict = red_acts.REDISPATCH_ACTIONS_DICT
        print(self.redispatch_actions_dict)
        # v1: observation size = powerflows + generators (prod_p) + (minute, hour, day and month)
        # v2: observation size = powerflows + loads (load_p) + generators (prod_p) + (minute, hour, day and month)
        # v3: observation size = powerflows + generators (prod_p) + generators (actual_d) + (minute, hour, day and month)
        self.observation_size = self.obs_space.n_line + self.obs_space.n_gen + 3
        # action size = decrease (max_ramp_down), stay or increase (max_ramp_up) dispatch for each redispatchable generator
        self.action_size = len(self.redispatch_actions_dict) #self.ACTIONS_PER_GEN ** sum(self.obs_space.gen_redispatchable)

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                        self.observation_size,
                                        num_frames = self.num_frames,
                                        learning_rate = self.lr,
                                        learning_rate_decay_steps = cfg.LR_DECAY_STEPS,
                                        learning_rate_decay_rate = cfg.LR_DECAY_RATE)
        
        # Setup training vars if needed
        if self.is_training:
            self._init_training()
        
    def convert_obs(self, observation):
        # v1: rho + prod_p + minutes + hour + day
        # v2: rho + load_p + prod_p + minutes + hour + day
        res = []

        if True:
            # include powerflow observations
            powerflow_limit = 1.6 # 200% of powerflow disconnects automatically the powerline
            max_val = powerflow_limit
            min_val = -powerflow_limit
            # new_val_x = min_new + (max_new - min_new) * (val_x - min_x) / (max_x - min_x)
            #pflow = -1 + 2 * (np.sign(observation.p_or) * observation.rho - min_val) / (max_val - min_val) 
            pflow = (np.sign(observation.p_or) * abs(observation.rho) - min_val) / (max_val - min_val) 
            res.append(pflow)
        
        if False:
            # include load_p observations
            max_sorted = np.sort(observation.gen_pmax)
            max_load = max_sorted[-1] + max_sorted[-2] # normalise by the two biggest generators
            load_norm = observation.load_p / max_load
            res.append(load_norm)
        
        if True:
            # include generation observations
            gen = observation.prod_p / (observation.gen_pmax * 1.1) # 1.1 (bef 1.1) because prod_p seems to go beyond the maximum
            res.append(gen)
        
        if False:
            # include actual dispatch
            max_disp = observation.gen_pmax
            min_disp = -observation.gen_pmax
            actual_d = (observation.actual_dispatch - min_disp) / (max_disp - min_disp) #0.01 + (1 - 0.01) *
            res.append(actual_d[observation.gen_redispatchable])

        if True:
            # include time (year not included, should work only on long term)
            res.append(np.array([observation.minute_of_hour / 60]))
            res.append(np.array([observation.hour_of_day / 24]))
            res.append(np.array([observation.day_of_week / 7]))
            #res.append(observation.month / 12)
        
        return np.concatenate(res)

    def convert_act(self, action):
        """
        Int action from my_act.
        Return a valid action
        """
        # nummber of actions: number of redispatches
        # 0: [(red_gen_id1, 0),(red_gen_id2, 0),(red_gen_id3, 0)]
        return self.action_space({"redispatch": self.redispatch_actions_dict[action]})
    
    def export_summary(self, log_path):
        file_path = self._find_file_in(log_path)
        event_acc = EventAccumulator(file_path)
        event_acc.Reload()
        cols = ['wtime', 'step', 'reward', 'alive', 'reward100', 'alive100', 'loss', 'lr']
        res = pd.DataFrame([])
        for metric in ['mean_reward', 'mean_alive', 'mean_reward_100', 'mean_alive_100', 'loss', 'lr']:
            if res.empty:
                res = pd.DataFrame([(w, s, tf.make_ndarray(t)) for w, s, t in event_acc.Tensors(metric)])
            else:
                res  = pd.concat([res, pd.DataFrame([(tf.make_ndarray(t)) for w, s, t in event_acc.Tensors(metric)])], axis=1)
        res.columns = cols
        res.to_csv(os.path.join(log_path, 'train_summary.csv'))
    
    def _find_file_in(self, path):
        files = os.listdir(path)
        fname = files[0] # there is only one file
        return os.path.join(path, fname)

class D3QNH(DoubleDuelingDQN):

    def __init__(self, observation_space, action_space, tph_ptdf=None, name=__name__, is_training=False, num_frames=4, batch_size=32, lr=1e-5):
        super().__init__(observation_space,
                        action_space,
                        name=name,
                        is_training=is_training)
        
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.lr = lr
        self.tph_ptdf = tph_ptdf

        self.ACTIONS_PER_GEN = 3
        self.redispatch_actions_dict = self._build_redispatch_dict()
        print(self.redispatch_actions_dict)
        
        # v1: observation size = powerflows + generators (prod_p) + (minute, hour, day and month)
        # v2: observation size = powerflows + loads (load_p) + generators (prod_p) + (minute, hour, day and month)
        # v3: observation size = powerflows + generators (prod_p) + generators (actual_d) + (minute, hour, day and month)
        self.observation_size = self.heuristic.get_line_observation_size() + self.obs_space.n_gen + sum(self.obs_space.gen_redispatchable) + 3
        
        self.action_size = self.ACTIONS_PER_GEN ** sum(self.obs_space.gen_redispatchable)

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                        self.observation_size,
                                        num_frames = self.num_frames,
                                        learning_rate = self.lr,
                                        learning_rate_decay_steps = cfg.LR_DECAY_STEPS,
                                        learning_rate_decay_rate = cfg.LR_DECAY_RATE)
        
        # Setup training vars if needed
        if self.is_training:
            self._init_training()
        

    def convert_obs(self, observation):
        ptdf = self.tph_ptdf.check_calculated_matrix(obs.line_status)
