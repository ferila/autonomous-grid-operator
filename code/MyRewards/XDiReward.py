import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float

# env.gen_cost_per_MW

class C3Di3Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.9)
        self.safety_redispatch_fct = dt_float(0.95)
        self.pf_threshold = dt_float(0.75)
        #self.reward_min = dt_float(-10*env.n_line - 10*env.n_gen)
        self.reward_min = dt_float(-400)
        self.reward_max = dt_float(0.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        #if has_error or is_illegal or is_ambiguous:
        #    return dt_float(-20)
        #else:
        # overflow lower than is_illegal
        #ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        #thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        #amp_ov = np.abs(ampere_flows - self.overflow_threshold*thermal_limits)
        
        relative_flow = env.backend.get_relative_flow()
        over_flow = np.abs(relative_flow) - self.overflow_threshold
        ovf_cost = dt_float(-200*np.sum(over_flow[over_flow > 0]))

        # active and reactive punishment
        prod_p, prod_q, _ = env.backend.generators_info()
        s_gen = np.sqrt(prod_p**2 + prod_q**2)
        pfactor = np.abs(prod_p / s_gen) - self.pf_threshold
        p_cost = dt_float(-200*np.sum(pfactor[pfactor > 0]))
        qfactor = np.abs(prod_q / s_gen) - self.pf_threshold
        q_cost = dt_float(-200*np.sum(qfactor[qfactor > 0]))

        # innecesary action (not donothing) better than is_illegal but worst than donothing
        # do_nothing reward 0 (it is preferable doing something than do not avoid overflows)
        #if action.as_dict():
        #    action_cost = dt_float(-5.0)
        #else:
        #    action_cost = dt_float(0.0)

        return ovf_cost + p_cost + q_cost

class C2Di3Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.9)
        self.safety_redispatch_fct = dt_float(0.95)
        #self.reward_min = dt_float(-10*env.n_line - 10*env.n_gen)
        self.reward_min = dt_float(-20)
        self.reward_max = dt_float(0.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        #if has_error or is_illegal or is_ambiguous:
        #    return dt_float(-20)
        #else:
        # overflow lower than is_illegal
        #ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        #thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        #amp_ov = np.abs(ampere_flows - self.overflow_threshold*thermal_limits)
        
        relative_flow = np.abs(env.backend.get_relative_flow(), dtype=dt_float)
        over_flow = (relative_flow - self.overflow_threshold)
        ovf_cost = dt_float(-20*np.sum(over_flow[over_flow > 0]))

        # generator punishment for taget_dispatch going beyond limits
        target_dispatch = env.get_obs().target_dispatch
        pmax = np.abs(env.gen_pmax, dtype=dt_float)
        pmin = np.abs(env.gen_pmin, dtype=dt_float)
        up_limit = target_dispatch/(pmax - pmin) - self.safety_redispatch_fct
        down_limit = target_dispatch/(pmax - pmin) + self.safety_redispatch_fct
        over_redisp_cost = dt_float(-10*np.sum(np.abs(up_limit[up_limit > 0])))
        under_redisp_cost = dt_float(-10*np.sum(np.abs(down_limit[down_limit < 0])))

        # innecesary action (not donothing) better than is_illegal but worst than donothing
        # do_nothing reward 0 (it is preferable doing something than do not avoid overflows)
        #if action.as_dict():
        #    action_cost = dt_float(-5.0)
        #else:
        #    action_cost = dt_float(0.0)

        return ovf_cost + over_redisp_cost + under_redisp_cost #+ action_cost

class CDi3Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.95)
        self.reward_min = dt_float(-50)
        self.reward_max = dt_float(0.0)
        self.safety_redispatch_fct = 0.95

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        #if has_error or is_illegal or is_ambiguous:
        #    return dt_float(-20)
        #else:
        # overflow lower than is_illegal
        #ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        #thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        #amp_ov = np.abs(ampere_flows - self.overflow_threshold*thermal_limits)
        
        relative_flow = np.abs(env.backend.get_relative_flow(), dtype=dt_float)
        over_flow = (relative_flow - self.overflow_threshold)
        ovf_cost = dt_float(-30*np.sum(over_flow[over_flow > 0]))

        # generator punishment for taget_dispatch going beyond limits
        target_dispatch = env.get_obs().target_dispatch
        pmax = np.abs(env.gen_pmax, dtype=dt_float)
        pmin = np.abs(env.gen_pmin, dtype=dt_float)
        up_limit = target_dispatch/(pmax - pmin) - self.safety_redispatch_fct
        down_limit = target_dispatch/(pmax - pmin) + self.safety_redispatch_fct
        over_redisp_cost = dt_float(-10*np.sum(np.abs(up_limit[up_limit > 0])))
        under_redisp_cost = dt_float(-10*np.sum(np.abs(down_limit[down_limit < 0])))

        # generator punishment for going beyond pmax
        prod_p, _, _ = env.backend.generators_info()
        up_gen = np.divide(prod_p, pmax, out=np.zeros_like(prod_p), where=pmax!=0) - 1
        down_gen = np.divide(pmin, prod_p, out=np.zeros_like(pmin), where=prod_p!=0) - 1
        over_prod_cost = dt_float(-10*np.sum(up_gen[up_gen > 0]))
        under_prod_cost = dt_float(-10*np.sum(down_gen[down_gen > 0]))

        # innecesary action (not donothing) better than is_illegal but worst than donothing
        # do_nothing reward 0 (it is preferable doing something than do not avoid overflows)
        #if action.as_dict():
        #    action_cost = dt_float(-5.0)
        #else:
        #    action_cost = dt_float(0.0)

        return ovf_cost + over_prod_cost + under_prod_cost + over_redisp_cost + under_redisp_cost #+ action_cost

class Di3Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.92)
        self.reward_min = dt_float(-20.0*env.n_line - 10*env.n_gen - 10*env.n_gen)
        self.reward_max = dt_float(0.0)
        self.safety_redispatch_fct = 0.95

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return dt_float(-20)
        else:
            # overflow lower than is_illegal
            #ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            #thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            #relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
            relative_flow = np.abs(env.backend.get_relative_flow(), dtype=dt_float)
            over_flow = (relative_flow - self.overflow_threshold)
            ovf_cost = dt_float(-50*np.sum(over_flow > 0))

            # generator punishment for taget_dispatch going beyond limits
            target_dispatch = env.get_obs().target_dispatch
            pmax = np.abs(env.gen_pmax, dtype=dt_float)
            pmin = np.abs(env.gen_pmin, dtype=dt_float)
            over_redisp_cost = dt_float(-10*np.sum((target_dispatch - self.safety_redispatch_fct*(pmax - pmin)) > 0))
            under_redisp_cost = dt_float(-10*np.sum((target_dispatch + self.safety_redispatch_fct*(pmax - pmin)) < 0))

            # generator punishment for going beyond pmax
            prod_p, _, _ = env.backend.generators_info()
            over_prod_cost = dt_float(-10*np.sum((prod_p - pmax) > 0))
            under_prod_cost = dt_float(-10*np.sum((pmin - prod_p) > 0))

            # innecesary action (not donothing) better than is_illegal but worst than donothing
            # do_nothing reward 0 (it is preferable doing something than do not avoid overflows)
            #if action.as_dict():
            #    action_cost = dt_float(-5.0)
            #else:
            #    action_cost = dt_float(0.0)

            return ovf_cost + over_prod_cost + under_prod_cost + over_redisp_cost + under_redisp_cost #+ action_cost

class CDi24Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.55) # 0.9 #default 0.95
        self.reward_min = dt_float(-60) #dt_float(-20.0*env.n_line - 10*env.n_gen - 5.0)
        self.reward_max = dt_float(0.0)
        self.pf_threshold = dt_float(0.75)
        self.pmaxmin_threshold = dt_float(0.9) #0.95

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        #if has_error or is_illegal or is_ambiguous:
        #    return dt_float(-10)
        #else:
        if is_illegal:
            leg_cost = -10
        else:
            leg_cost = 0

        # overflow lower than is_illegal
        #ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        #thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        #relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
        rho = np.abs(env.backend.get_relative_flow())
        over_flow = np.maximum(rho - self.overflow_threshold, 0)
        ovf_cost = dt_float(-70*np.sum(over_flow**2))

        # active and reactive punishment
        prod_p, prod_q, _ = env.backend.generators_info()
        prod_p = prod_p[[True,True,False,False,False,True]]
        prod_q = prod_q[[True,True,False,False,False,True]]
        s_gen = np.sqrt(prod_p**2 + prod_q**2)
        pfactor = np.maximum(np.abs(prod_p / s_gen) - self.pf_threshold, 0)
        p_cost = 0#dt_float(-30*np.sum(pfactor**2))
        qfactor = np.maximum(np.abs(prod_q / s_gen) - self.pf_threshold, 0)
        q_cost = 0#dt_float(-30*np.sum(qfactor**2))

        # pmax limits
        pmax = np.abs(env.gen_pmax)[[True,True,False,False,False,True]]
        pmin = np.abs(env.gen_pmin)[[True,True,False,False,False,True]]
        over = np.maximum(prod_p/pmax - self.pmaxmin_threshold, 0)
        over_redisp_cost = dt_float(-30*np.sum(over**2))
        under = np.maximum((pmin - prod_p)/pmax + 1 - self.pmaxmin_threshold, 0)        
        under_redisp_cost = dt_float(-30*np.sum(under**2))

        return leg_cost + ovf_cost + p_cost + q_cost + over_redisp_cost + under_redisp_cost

class Di24Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.9) #default 0.95
        self.reward_min = dt_float(-20.0*env.n_line - 10*env.n_gen - 5.0)
        self.reward_max = dt_float(0.0)
        self.pf_threshold = dt_float(0.75)
        self.pmaxmin_threshold = dt_float(0.95)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        #if has_error or is_illegal or is_ambiguous:
        #    return dt_float(-10)
        #else:
        if is_illegal:
            leg_cost = -10
        else:
            leg_cost = 0

        # overflow lower than is_illegal
        #ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        #thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        #relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
        rho = np.abs(env.backend.get_relative_flow())
        over_flow = (rho - self.overflow_threshold)
        ovf_cost = dt_float(-30*np.sum(over_flow > 0))

        # active and reactive punishment
        prod_p, prod_q, _ = env.backend.generators_info()
        prod_p = prod_p[[True,True,False,False,False,True]]
        prod_q = prod_q[[True,True,False,False,False,True]]
        s_gen = np.sqrt(prod_p**2 + prod_q**2)
        pfactor = np.abs(prod_p / s_gen) - self.pf_threshold
        p_cost = dt_float(-30*np.sum(pfactor > 0))
        qfactor = np.abs(prod_q / s_gen) - self.pf_threshold
        q_cost = dt_float(-30*np.sum(qfactor > 0))

        # pmax limits
        pmax = np.abs(env.gen_pmax, dtype=dt_float)[[True,True,False,False,False,True]]
        pmin = np.abs(env.gen_pmin, dtype=dt_float)[[True,True,False,False,False,True]]
        over = prod_p - pmax * self.pmaxmin_threshold
        over_redisp_cost = dt_float(-30*np.sum(over > 0))
        under = pmin + pmax*(1-self.pmaxmin_threshold) - prod_p
        under_redisp_cost = dt_float(-30*np.sum(under > 0))

        return leg_cost + ovf_cost + p_cost + q_cost + over_redisp_cost + under_redisp_cost

class Di23Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.9) #default 0.95
        self.reward_min = dt_float(-20.0*env.n_line - 10*env.n_gen - 5.0)
        self.reward_max = dt_float(0.0)
        self.pf_threshold = dt_float(0.75)
        self.safety_redispatch_fct = dt_float(0.9)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return dt_float(-10)
        else:
            # overflow lower than is_illegal
            ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
            over_flow = (relative_flow - self.overflow_threshold)
            ovf_cost = dt_float(-20*np.sum(over_flow > 0))

            # active and reactive punishment
            prod_p, prod_q, _ = env.backend.generators_info()
            s_gen = np.sqrt(prod_p**2 + prod_q**2)
            pfactor = np.abs(prod_p / s_gen) - self.pf_threshold
            p_cost = dt_float(-30*np.sum(pfactor > 0))
            qfactor = np.abs(prod_q / s_gen) - self.pf_threshold
            q_cost = dt_float(-30*np.sum(qfactor > 0))

            # target_dispatch limits
            target_dispatch = env.get_obs().target_dispatch
            pmax = np.abs(env.gen_pmax, dtype=dt_float)
            pmin = np.abs(env.gen_pmin, dtype=dt_float)
            over = target_dispatch - self.safety_redispatch_fct*(pmax - pmin)
            over_redisp_cost = dt_float(-30*np.sum(over > 0))
            under = target_dispatch + self.safety_redispatch_fct*(pmax - pmin)
            under_redisp_cost = dt_float(-30*np.sum(under < 0))

            return ovf_cost + p_cost + q_cost + over_redisp_cost + under_redisp_cost

class Di22Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.95) #default 0.95
        self.reward_min = dt_float(-20.0*env.n_line - 10*env.n_gen - 5.0)
        self.reward_max = dt_float(0.0)
        self.pf_threshold = dt_float(0.75)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return dt_float(-10)
        else:
            # overflow lower than is_illegal
            ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
            over_flow = (relative_flow - self.overflow_threshold)
            ovf_cost = dt_float(-20*np.sum(over_flow > 0))

            # active and reactive punishment
            prod_p, prod_q, _ = env.backend.generators_info()
            s_gen = np.sqrt(prod_p**2 + prod_q**2)
            pfactor = np.abs(prod_p / s_gen) - self.pf_threshold
            p_cost = dt_float(-30*np.sum(pfactor > 0))
            qfactor = np.abs(prod_q / s_gen) - self.pf_threshold
            q_cost = dt_float(-30*np.sum(qfactor > 0))

            return ovf_cost + p_cost + q_cost

class Di2Reward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.95) #default 0.95
        self.reward_min = dt_float(-20.0*env.n_line - 10*env.n_gen - 5.0)
        self.reward_max = dt_float(0.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return dt_float(-10)
        else:
            # overflow lower than is_illegal
            ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
            over_flow = (relative_flow - self.overflow_threshold)
            ovf_cost = dt_float(-20*np.sum(over_flow > 0))

            # generator punishment for going beyond pmax
            prod_p, _, _ = env.backend.generators_info()
            pmax = np.abs(env.gen_pmax, dtype=dt_float)
            pmin = np.abs(env.gen_pmin, dtype=dt_float)
            over_prod_cost = dt_float(-10*np.sum((prod_p - pmax) > 0))
            under_prod_cost = dt_float(-10*np.sum((pmin - prod_p) > 0))

            # innecesary action (not donothing) better than is_illegal but worst than donothing
            # do_nothing reward 0 (it is preferable doing something than do not avoid overflows)
            #if action.as_dict():
            #    action_cost = dt_float(-5.0)
            #else:
            #    action_cost = dt_float(0.0)

            return ovf_cost + over_prod_cost + under_prod_cost #+ action_cost

class FoolReward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(10.0)
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if action.as_dict():
            return self.reward_min
        else:
            return self.reward_max

class DiReward(BaseReward):

    def __init__(self):
        BaseReward.__init__(self)
    
    def initialize(self, env):
        # overflow threshold to act
        self.overflow_threshold = dt_float(0.95)
        self.reward_min = dt_float(-20.0*env.n_line - 1.0)
        self.reward_max = dt_float(0.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return dt_float(-10)
        else:
            # overflow lower than is_illegal
            ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
            over_flow = (relative_flow - self.overflow_threshold)
            ovf_cost = dt_float(-20*np.sum(over_flow > 0))

            # innecesary action (not donothing) better than is_illegal but worst than donothing
            # do_nothing reward 0 (it is preferable doing something than do not avoid overflows)
            if action.as_dict():
                action_cost = dt_float(-5.0)
            else:
                action_cost = dt_float(0.0)

            return ovf_cost + action_cost

class OvdiReward(BaseReward):
    """
    This reward can be used for environments where redispatching is availble. 
    It assigns a cost to redispatching action, penalizes with the losses and penalises heavily overflow
    """
    def __init__(self, alpha_redisph=5.0):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.max_regret = dt_float(0.0)
        self.alpha_redisph = dt_float(alpha_redisph)
        self.overflow_punishment = 100

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException("Impossible to use the RedispReward reward with an environment without generators"
                                   "cost. Please make sure env.redispatching_unit_commitment_availble is available.")
        worst_marginal_cost = np.max(env.gen_cost_per_MW)
        worst_load = dt_float(np.sum(env.gen_pmax))
        worst_losses = dt_float(0.05) * worst_load  # it's not the worst, but definitely an upper bound
        worst_redisp = self.alpha_redisph * np.sum(env.gen_pmax)  # not realistic, but an upper bound
        worst_overflow_ts = np.sum(self.overflow_punishment**np.array([2,1]) - 1)
        self.max_regret = (worst_losses + worst_redisp)*worst_marginal_cost + worst_overflow_ts
        self.reward_min = dt_float(-50.0) # @felipe: changed from -10

        least_loads = dt_float(worst_load * 0.5)  # half the capacity of the grid
        least_losses = dt_float(0.015 * least_loads)  # 1.5% of losses
        least_redisp = dt_float(0.0)  # lower_bound is 0
        base_marginal_cost = np.min(env.gen_cost_per_MW[env.gen_cost_per_MW > 0.])
        min_regret = (least_losses + least_redisp) * base_marginal_cost
        self.reward_max = dt_float((self.max_regret - min_regret) / least_loads)

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            res = self.reward_min
        else:
            # compute the losses
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            losses = np.sum(gen_p) - np.sum(load_p)
            obs = env.get_obs()
            
            # compute overflow cost
            overflow_cost = np.sum(self.overflow_punishment**obs.timestep_overflow - 1)

            # compute the marginal cost
            marginal_cost = np.max(env.gen_cost_per_MW[env.gen_activeprod_t > 0.])

            # redispatching amount
            redisp_cost = self.alpha_redisph * np.sum(np.abs(env.actual_dispatch)) * marginal_cost

            # cost of losses
            losses_cost = losses * marginal_cost

            # total "regret"
            regret = losses_cost + redisp_cost + overflow_cost

            # compute reward
            reward = self.max_regret - regret

            # divide it by load, to be less sensitive to load variation
            res = dt_float(reward / np.sum(load_p))

        return res


class LidiReward(BaseReward):
    """
    This reward can be used for environments where redispatching is availble. 
    It assigns a cost to redispatching action, penalizes with the losses and assign lines usage cost
    """
    def __init__(self, alpha_redisph=5.0):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.max_regret = dt_float(0.0)
        self.alpha_redisph = dt_float(alpha_redisph)

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException("Impossible to use the RedispReward reward with an environment without generators"
                                   "cost. Please make sure env.redispatching_unit_commitment_availble is available.")
        worst_marginal_cost = np.max(env.gen_cost_per_MW)
        worst_load = dt_float(np.sum(env.gen_pmax))
        worst_losses = dt_float(0.05) * worst_load  # it's not the worst, but definitely an upper bound
        worst_redisp = self.alpha_redisph * np.sum(env.gen_pmax)  # not realistic, but an upper bound

        lines_limit = np.sum(np.abs(env.backend.get_thermal_limit()))
        worst_line_usage = self.alpha_redisph * lines_limit/(100*env.n_line) # it is just a proxy
        
        self.max_regret = (worst_losses + worst_redisp + worst_line_usage)*worst_marginal_cost
        self.reward_min = dt_float(-10.0)

        least_loads = dt_float(worst_load * 0.5)  # half the capacity of the grid
        least_losses = dt_float(0.015 * least_loads)  # 1.5% of losses
        least_redisp = dt_float(0.0)  # lower_bound is 0
        least_line_usage = dt_float(0.0) # 0 is no line usage
        base_marginal_cost = np.min(env.gen_cost_per_MW[env.gen_cost_per_MW > 0.])
        min_regret = (least_losses + least_redisp + least_line_usage) * base_marginal_cost
        self.reward_max = dt_float((self.max_regret - min_regret) / least_loads)

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            res = self.reward_min
        else:
            # compute the losses
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            flow_p, *_ = env.backend.lines_or_info()
            losses = np.sum(gen_p) - np.sum(load_p)
            line_usage = np.sum(np.abs(flow_p)) / (100*env.n_line)

            # compute the marginal cost
            marginal_cost = np.max(env.gen_cost_per_MW[env.gen_activeprod_t > 0.])

            # redispatching amount
            redisp_cost = self.alpha_redisph * np.sum(np.abs(env.actual_dispatch)) * marginal_cost

            # cost of losses
            losses_cost = losses * marginal_cost

            # gain of low line usage
            line_usage_cost = self.alpha_redisph * line_usage * marginal_cost

            # total "regret"
            regret = losses_cost + redisp_cost + line_usage_cost

            # compute reward
            reward = self.max_regret - regret

            # divide it by load, to be less sensitive to load variation
            res = dt_float(reward / np.sum(load_p))

        return res