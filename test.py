import os
import sys
import pdb
import grid2op
import numpy as np
from tqdm import tqdm
# from tqdm.notebook import tqdm
from grid2op.Agent import DoNothingAgent, BaseAgent


# Redispatching

class GreedyEconomic(BaseAgent):
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)
        self.do_nothing = action_space()
        
    def act(self, obs, reward, done):
        act = self.do_nothing
        if obs.prod_p[0] < obs.gen_pmax[0] - 1 and \
        obs.target_dispatch[0] < (obs.gen_pmax[0] - obs.gen_max_ramp_up[0]) - 1 and\
        obs.prod_p[0] > 0.:
            # if the cheapest generator is significantly bellow its maximum cost
            if obs.target_dispatch[0] < obs.gen_pmax[0]:
                #in theory i can still ask for more
                act = env.action_space({"redispatch": [(0, obs.gen_max_ramp_up[0])]})
        return act

if __name__ == "__main__":
    max_iter = 100

    env = grid2op.make("case14_redisp",
                        chronics_path="D:\\ESDA_MSc\\Dissertation\\code_stuff\\case14_redisp")
    # print("Is this environment suitable for redispatching: {}".format(env.redispatching_unit_commitment_availble))

    '''It has one solar and one wind generator (that cannot be dispatched), 
    one nuclear powerplant (dispatchable) 
    and 2 thermal generators (dispatchable also). 
    This problem is then a problem of continuous control with 3 degress of freedom.
    '''

    agent = GreedyEconomic(env.action_space)
    done = False
    reward = env.reward_range[0]

    env.set_id(0) # reset the env to the same id
    obs = env.reset()
    cum_reward = 0
    nrow = env.chronics_handler.max_timestep() if max_iter <= 0 else max_iter
    gen_p = np.zeros((nrow, env.n_gen))
    gen_p_setpoint = np.zeros((nrow, env.n_gen))
    load_p = np.zeros((nrow, env.n_load))
    rho = np.zeros((nrow, env.n_line))
    i = 0
    with tqdm(total=max_iter, desc="step") as pbar:
        while not done:
            act = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)
    #         print("act: {}".format(act))
    #         print("info: {}".format(info['exception']))
    #         if info['exception'] is not None:
            if np.abs(np.sum(obs.actual_dispatch)) > 1e-2:
                pdb.set_trace()
            data_generator = env.chronics_handler.real_data.data
            gen_p_setpoint[i,:] = data_generator.prod_p[data_generator.current_index, :]
            gen_p[i,:] = obs.prod_p
            load_p[i,:] = obs.load_p
            rho[i,:] = obs.rho
            cum_reward += reward
            i += 1
            pbar.update(1)
            if i >= max_iter:
                break
    print("The cumulative reward with this agent is {:.0f}".format(cum_reward))