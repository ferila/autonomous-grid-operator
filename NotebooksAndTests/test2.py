import sys
import grid2op
import numpy as np
import matplotlib.pyplot as plt
from grid2op.Agent import DoNothingAgent

#env = grid2op.make("case14_redisp", chronics_path="D:\\ESDA_MSc\\Dissertation\\code_stuff\\case14_redisp")
env = grid2op.make("rte_case14_redisp")
#env = grid2op.make("l2rpn_case14_sandbox")

# do_nothing_act = env.helper_action_player({})
# obs, reward, done, info = env.step(do_nothing_act)

agent = DoNothingAgent(env.action_space) # RedispReward
# env.gen_cost_per_MW: marginal cost

obs = env.reset()
rew = env.reward_range[0]
done = False

# obs.to_vect(): all_info
# remember an element is either an end of a powerline, or a generator or a load
# main difference: simulation
#   do_nothing_act = env.helper_action_player({})
#   obs_sim, reward_sim, is_done_sim, info_sim = obs.simulate(do_nothing_act)

print("Number of generators of the powergrid: {}".format(obs.n_gen))
print("Number of loads of the powergrid: {}".format(obs.n_load))
print("Number of powerline of the powergrid: {}".format(obs.n_line))
print("Number of elements connected to each substations in the powergrid: {}".format(obs.sub_info))
print("Total number of elements: {}".format(obs.dim_topo))

print("matrix elements size: {}".format(obs.connectivity_matrix().shape))

print("Redispatchables/max_ramp: \n{}".format(env.gen_redispatchable))
print(env.gen_min_uptime) #[96  4  0  0  4]
print(env.gen_max_ramp_up) #[ 5. 10.  0.  0. 10.]
print(env.gen_pmax) #[150. 200.  70.  50. 300.]
print(env.gen_pmin) #
gen = []
load = []
#max_iter = 2000
it = 0
#print("Initial (weekday {}) --- {}/{}/{} {}:{}------".format(obs.day_of_week, obs.day, obs.month, obs.year, obs.hour_of_day, obs.minute_of_hour, obs.day_of_week))
sys.exit(0)
while not done:
    print("-----step {} (day {}) --- {}/{}/{} {}:{}------".format(it, obs.day_of_week, obs.day, obs.month, obs.year, obs.hour_of_day, obs.minute_of_hour, obs.day_of_week))
    print("Total prod: {}".format(np.sum(obs.prod_p)))
    print("Total load: {}".format(np.sum(obs.load_p)))
    
    if it < 500:
        act = agent.act(obs, rew, done)
    else:
        act = env.action_space({"redispatch": [(4,+10)]})
    print(act.is_ambiguous())
    obs, rew, done, info = env.step(act)
    #if it >= 500:
    #    print("dispatch: {}".format(obs.actual_dispatch))
    #    print("Tot prod/load: \n{}\n{}".format(np.sum(obs.prod_p), np.sum(obs.load_p)))

    gen.append(obs.prod_p)
    load.append(obs.load_p)

    #print("Production: \n{}".format(obs.prod_p))
    #print("Load: \n{}".format(obs.load_p))
    #print("Rho: \n{}".format(obs.rho))
    #if it >= max_iter:
    #    done = True
    #    print("Ended by max_iter")
    it += 1

#print("Initial (weekday {}) --- {}/{}/{} {}:{}------".format(obs.day_of_week, obs.day, obs.month, obs.year, obs.hour_of_day, obs.minute_of_hour, obs.day_of_week))
gen = np.array(gen)

fig, axs = plt.subplots(6,1)

axs[0].plot(range(len(gen)), np.sum(gen, axis=1))
axs[0].plot(range(len(load)), np.sum(load, axis=1))

for i in range(obs.n_gen):
    axs[i+1].plot(range(len(gen)), gen[:,i], label=env.gen_type[i])
    axs[i+1].axhline(y=env.gen_pmax[i], color='r')
    axs[i+1].axhline(y=env.gen_pmin[i], color='r')
    axs[i+1].legend()

plt.show()

