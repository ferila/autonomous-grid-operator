#from DQN_NNb import DQN_NNb
import os
import numpy as np
import pandas as pd
from l2rpn_baselines.DoubleDuelingDQN import DoubleDuelingDQN
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class MyDoubleDuelingDQN(DoubleDuelingDQN):

    def __init__(self, observation_space, action_space, name=__name__, is_training=False, num_frames=4, batch_size=32, lr=1e-5):
        super().__init__(observation_space,
                        action_space,
                        name=name,
                        is_training=is_training)
        
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.lr = lr

        self.ACTIONS_PER_GEN = 3
        self.redispatch_actions_dict = self._build_redispatch_dict()
        # observation size = powerflows + generators + (minute, hour, day and month)
        self.observation_size = self.obs_space.n_line + self.obs_space.n_gen + 3
        # action size = decrease (max_ramp_down), stay or increase (max_ramp_up) dispatch for each redispatchable generator
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

    def _build_redispatch_dict(self):     
        res_dict = {}
        gen_action_ixs = [0 for i in range(sum(self.obs_space.gen_redispatchable))]
        for i in range(self.action_size):
            res_dict[i] = self._add_group_action(gen_action_ixs)
            gen_action_ixs = self._change_action_ixs(i, gen_action_ixs)
        return res_dict

    def _add_group_action(self, gen_action):
        redis_gen_ids = np.array(range(self.obs_space.n_gen))[self.obs_space.gen_redispatchable]
        one_action = []
        for i, g_id in enumerate(redis_gen_ids):
            act_g = self._get_gen_action(g_id, gen_action[i])
            one_action.append((g_id, act_g))
        return one_action
    
    def _get_gen_action(self, gen_id, action_ix):
        """
        Define the granularity of actions. Intermediated values could be added.
        If add here, change parameter self.ACTIONS_PER_GEN
        """
        if action_ix == 0:
            return -self.obs_space.gen_max_ramp_down[gen_id]
        elif action_ix == 1:
            return 0
        elif action_ix == 2:
            return self.obs_space.gen_max_ramp_up[gen_id]
        else:
            raise Exception("Action index out of bounds")

    def _change_action_ixs(self, act_ix, gen_action_ixs):
        """
        cont_g_ix tells which generator is changing.
        gen_action_ixs contains action index for each generator.
        """
        limits = [self.ACTIONS_PER_GEN ** i for i in range(sum(self.obs_space.gen_redispatchable))]
        # [1, 3, 9]
        
        for ix, gen_lim in enumerate(limits):
            if (act_ix + 1) % gen_lim == 0:
                if gen_action_ixs[ix] == (self.ACTIONS_PER_GEN - 1):
                    gen_action_ixs[ix] = 0
                else:
                    gen_action_ixs[ix] += 1

        return gen_action_ixs

        
    def convert_obs(self, observation):
        res = []
        
        # include generation observations
        gen = observation.prod_p / (observation.gen_pmax * 1.1) #1.1 because prod_p seems to go beyond the maximum
        res.append(gen)

        # include powerflow observations
        powerflow_limit = 2 # 200% of powerflow disconnects automatically the powerline
        max_val = powerflow_limit
        min_val = -powerflow_limit
        pflow = (np.sign(observation.p_or) * observation.rho - min_val) / (max_val - min_val)
        res.append(pflow)

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
        res = pd.DataFrame()
        for metric in ['mean_reward', 'mean_alive', 'mean_reward_100', 'mean_alive_100', 'loss', 'lr']:
            if res.empty:
                res = pd.DataFrame([(w, s, tf.make_ndarray(t)) for w, s, t in event_acc.Tensors(metric)])
            else:
                res  = pd.concat([res, pd.DataFrame([(tf.make_ndarray(t)) for w, s, t in event_acc.Tensors(metric)])], axis=1)
        res.columns = cols
        res.to_csv(os.path.join(log_path, 'summary'))
    
    def _find_file_in(self, path):
        files = os.listdir(path)
        fname = files[0] # there is only one file
        return os.path.join(path, fname)
