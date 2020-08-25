import numpy as np

class RedispatchActions(object):

    def __init__(self, observation_space, action_space, acts_per_gen=3, max_setpoint_change=5):
        self.obs_space = observation_space
        self.act_space = action_space
        # actions allowed per generator
        self.ACTIONS_PER_GEN = acts_per_gen
        # change in the geneartion group setpoint
        self.MAXIMUM_SETPOINT_CHANGE = max_setpoint_change
        self.REDISPATCH_ACTIONS_DICT = self._build_redispatch_dict()
    
    def _build_redispatch_dict(self):     
        res_dict = {}
        gen_action_ixs = [0 for i in range(sum(self.obs_space.gen_redispatchable))]
        possible_actions = self.ACTIONS_PER_GEN ** sum(self.obs_space.gen_redispatchable)
        cont = 0
        for i in range(possible_actions):
            candidate_group_action = self._get_group_action(gen_action_ixs)
            balance_generation = sum([a[1] for a in candidate_group_action])
            # As current dispatch already balance demand (setpoint), 
            # the idea is to not change the total setpoint
            if abs(balance_generation) <= self.MAXIMUM_SETPOINT_CHANGE:
                res_dict[cont] = candidate_group_action
                cont += 1
            gen_action_ixs = self._change_action_ixs(i, gen_action_ixs)
        return res_dict

    def _get_group_action(self, gen_action):
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
