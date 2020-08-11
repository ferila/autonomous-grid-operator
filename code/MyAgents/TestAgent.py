import sys
import copy
import grid2op
import numpy as np
import pandas as pd
import pandapower as pp

from grid2op.Runner import Runner
from grid2op.Agent import DoNothingAgent
from grid2op.Agent import BaseAgent
from grid2op.Agent import GreedyAgent
from grid2op.Agent import AgentWithConverter
from grid2op.Reward import BaseReward

import copy
from grid2op.PlotGrid import PlotMatplot
from grid2op.Episode import EpisodeData

pd.options.display.max_columns = 20

class PTDFBased(BaseAgent):
    
    def __init__(self, action_space, ptdf):
        BaseAgent.__init__(self, action_space)
        # has self.action_space

    def act(self, observation, reward, done=False):
        observation.prod_p

        # get generation by node

        # apply to ptdf and see dispatches

        # act over dispatch if some line is reaching limit


class simpleAgent2(AgentWithConverter):

    def __init__(self, action_space):
        AgentWithConverter.__init__(self, action_space)
    
    def convert_obs(self, observation):
        # take observation.prod_p, observation.rho and do something like concatenate
        pass

    def convert_act(self, transformed_action):
        # take the output of the "special" policy and decoded to known action_space values
        # act = self.action_space({"set_status": transformed_action})
        pass

    def my_act(self, transformed_observation, reward, done=False):
        # do the "special" policy
        # transformed_action = self.my_neuronal_network(transformed_observation)
        transformed_action = None
        return transformed_action

#Action selection strategies:
# - greedy
# - e-greedy
# - softmax (Boltzman distribution)
# - choose state with probability p(s) = V(s)/Sum(V(x),x)

class simpleTxGreedy(GreedyAgent):
    def __init__(self, action_space):
        GreedyAgent.__init__(self, action_space)
        # self.the_actions = None

    def _get_tested_action(self, observation):
        res = [self.action_space({})]

        # those who manage the more intensive line loss
        id_intensive_line = np.argmax(observation.rho)

        # those who manage the bigger operative generator loss
        # id_intensive_generator = np.argmax(observation.prod_p)

        return res

#class simpleReward(BaseReward):
#    pass

"""Example from notebooks"""
class GreedyEconomic(BaseAgent): 
    def __init__(self, action_space):
        super().__init__(action_space)
        self.do_nothing = action_space()
        
    def act(self, obs, reward, done):
        act = self.do_nothing
        if obs.prod_p[0] < obs.gen_pmax[0] - 1 and \
        obs.target_dispatch[0] < (obs.gen_pmax[0] - obs.gen_max_ramp_up[0]) - 1 and\
        obs.prod_p[0] > 0.:
            # if the cheapest generator is significantly bellow its maximum cost
            if obs.target_dispatch[0] < obs.gen_pmax[0]:
                #in theory i can still ask for more
                act = self.action_space({"redispatch": [(0, obs.gen_max_ramp_up[0])]})
        return act
