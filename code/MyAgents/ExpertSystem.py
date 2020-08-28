import sys
import copy
import numpy as np
from grid2op.Agent import BaseAgent
from .Actions import RedispatchActions, RedispatchSpecificActions

class ExpertSystem(BaseAgent):
    def __init__(self, observation_space, action_space, tph_ptdf):
        super().__init__(action_space)
        self.tph_ptdf = tph_ptdf
        self.overflow_threshold = 0.95
        self.obs_space = observation_space
        red_acts = RedispatchSpecificActions(observation_space, action_space, max_setpoint_change=0)
        self.redispatch_actions_dict = red_acts.REDISPATCH_ACTIONS_DICT
    
    def act(self, obs, reward, done):
        # act only if flows are over its limits
        if np.sum(np.abs(obs.rho) > self.overflow_threshold):
            line_ix = np.argmax(np.abs(obs.rho))
            best_action = self._analyse_dispatch_options(obs, line_ix)
            act = self.action_space({"redispatch": self.redispatch_actions_dict[best_action]})
        else:
            act = self.action_space({})
        return act
    
    def _analyse_dispatch_options(self, obs, line_ix):
        """
        select best action among the availables in redispatch_action_dict
        """

        # calculate current scenario
        ptdf_matrix = self._check_topology_change(obs)
        current_net_gen_sub = self.__gen_by_sub(obs)
        ptdf_current_flow = ptdf_matrix.loc[line_ix,:].dot(current_net_gen_sub)
        overflow = np.abs(obs.rho[line_ix]) - self.overflow_threshold

        lower_target_flow = 1000

        for act, redisp in self.redispatch_actions_dict.items():
            next_net_gen_sub = self.__add_power(current_net_gen_sub, redisp, obs)
            #predicted_flow = ptdf_matrix.dot(next_net_gen_sub)
            target_line_flow = abs(ptdf_matrix.loc[line_ix, :].dot(next_net_gen_sub) / (ptdf_current_flow * (1 - overflow)))
            #has_overflows = self.__check_overflows(predicted_flow)
            if target_line_flow < lower_target_flow: # and not has_overflows:
                best_action = act
                lower_target_flow = target_line_flow
        
        return best_action

    def _check_topology_change(self, obs):
        """
        It checks if ptdf matrix is already created for the actual operational lines.
        Return: ptdf matrix
        """
        return self.tph_ptdf.check_calculated_matrix(obs.line_status)

    @staticmethod
    def __gen_by_sub(observation):
        """
        Return an array with the total generation of each bus
        """
        total_gen = np.zeros(observation.n_sub)
        for i, g in enumerate(observation.prod_p):
            sub = observation.gen_to_subid[i]
            total_gen[sub] += g
        for i, l in enumerate(observation.load_p):
            sub = observation.load_to_subid[i]
            total_gen[sub] -= l

        return total_gen
    
    @staticmethod
    def __add_power(current, redispatch, observation):
        for g_ix, redisp_pw in redispatch:
            sub_ix = observation.gen_to_subid[g_ix]
            current[sub_ix] += redisp_pw
        return current
    
    @staticmethod
    def __check_overflows(flows):
        ans = False
        return ans

class MyFirstAgent(BaseAgent):
    """
    Deprecated
    """
    def __init__(self, observation_space, action_space, tph_ptdf):
        raise Exception("Class deprecated, use ExpertSystem instead.")
        super().__init__(action_space)
        self.tph_ptdf = tph_ptdf
        self.obs_space = observation_space
        self.act_space = action_space
        red_acts = RedispatchActions(observation_space, action_space)
        self.redispatch_actions_dict = red_acts.REDISPATCH_ACTIONS_DICT
    
    def act(self, obs, reward, done):
        # act only if flows are over its limits
        if np.sum(obs.rho > 1):
            line_ix = np.argmax(obs.rho)
            _, (best_gen, direction) = self._analyse_dispatch_options(obs, line_ix)
            # print("evauluations: \n{}".format(evaluations))
            # print("best gen {} (sub {}), direction: {}".format(best_gen, obs.gen_to_subid[best_gen], direction))
            if direction == 0:
                act = self.action_space({"redispatch": [(best_gen, -obs.gen_max_ramp_down[best_gen])]})
            elif direction == 1:
                act = self.action_space({"redispatch": [(best_gen, obs.gen_max_ramp_up[best_gen])]})
        else:
            act = self.action_space({})
        return act     

    def _analyse_dispatch_options(self, obs, line_ix):
        """
        Analyse the current flow and possible flows after redispatching each generators
        Uses ptdf to analyse impact of redispatching
        Returns: dictionary(generator_id, 0/1) 0: down, 1: up
        """
        ptdf_matrix = self._check_topology_change(obs)

        current_net_gen_sub = self._gen_by_sub(obs)
        ptdf_current_flow = ptdf_matrix.loc[line_ix, :].dot(current_net_gen_sub)
        
        #current_flow = obs.p_or[line_ix] #+ ptdf_matrix.loc[line_ix, :].dot(current_net_gen_sub)
        #from_sub = obs.line_or_to_subid[line_ix]
        #to_sub = obs.line_ex_to_subid[line_ix]
        # print("---current flow line {} ({}->{}): {}".format(line_ix, from_sub, to_sub, current_flow))
        # print("---compared to ptdf: {}".format(ptdf_current_flow))

        evaluated_flow = {}
        lower_flow = 1 #abs(ptdf_current_flow) #abs(current_flow)
        overflow = obs.rho[line_ix] - 1
        # print("should be: {}".format(overflow))
        redispatchables = np.array(range(obs.n_gen))[obs.gen_redispatchable]

        # This for could be replaced by a matrix calculation but ble
        for g in redispatchables:
            sub = obs.gen_to_subid[g]

            down = copy.deepcopy(current_net_gen_sub)
            down[sub] = current_net_gen_sub[sub] - obs.gen_max_ramp_down[g]
            evaluated_flow[g,0] = ptdf_matrix.loc[line_ix, :].dot(down) / (ptdf_current_flow * (1-overflow))

            up = copy.deepcopy(current_net_gen_sub)
            up[sub] = current_net_gen_sub[sub] + obs.gen_max_ramp_up[g]
            evaluated_flow[g,1] = ptdf_matrix.loc[line_ix, :].dot(up) / (ptdf_current_flow * (1-overflow))

            if abs(evaluated_flow[g, 0]) < lower_flow:
                best_g = g 
                direction = 0
                lower_flow = abs(evaluated_flow[g, 0])
            if abs(evaluated_flow[g, 1]) < lower_flow:
                best_g = g
                direction = 1
                lower_flow = abs(evaluated_flow[g ,1])

        #abs(ptdf_current_flow) - abs(evaluated_flow[g, 0]) < abs(ptdf_current_flow) * overflow_perc

        return evaluated_flow, (best_g, direction)

    def _check_topology_change(self, obs):
        """
        It checks if ptdf matrix is already created for the actual operational lines.
        Return: ptdf matrix
        """
        return self.tph_ptdf.check_calculated_matrix(obs.line_status)

    def _gen_by_sub(self, observation):
        """
        Return an array with the total generation of each bus
        """
        total_gen = np.zeros(observation.n_sub)
        for i, g in enumerate(observation.prod_p):
            sub = observation.gen_to_subid[i]
            total_gen[sub] += g
        for i, l in enumerate(observation.load_p):
            sub = observation.load_to_subid[i]
            total_gen[sub] -= l

        return total_gen