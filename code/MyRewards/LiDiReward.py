import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float

class LidiReward(BaseReward):
    """
    This reward can be used for environments where redispatching is availble. 
    It assigns a cost to redispatching action, penalizes with the losses and lines usage
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