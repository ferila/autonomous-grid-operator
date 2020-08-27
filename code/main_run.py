import os
import time
import shutil
import grid2op
from MyRunner import MyRunner
from Heuristic import TopologyPTDF
from grid2op.Reward import L2RPNReward, RedispReward
from MyRewards.XDiReward import LidiReward, OvdiReward
from grid2op.Agent import DoNothingAgent
from MyAgents.DQNb import D3QN, D3QNH
from MyAgents.ExpertSystem import ExpertSystem
#from l2rpn_baselines.DoubleDuelingDQN import train as DDDQN_train

# tZero: normal (powerflow (0,1) without load)
# tOne: with load
# tTwo: powerflow from (0,1) to (-1,1)
# tThree: generation plus actual dispatch

# all with acts reduced
# y: l2rpn reward
# yR: redisp reward
# yT: lidi reward

if __name__ == "__main__":
    reward_to_use = L2RPNReward #GameplayReward in "l2rpn_wcci_2020"
    train_iter = 25000
    for env_case_name in ["l2rpn_case14_sandbox"]: # "rte_case14_redisp" # rte_case14_redisp, l2rpn_case14_sandbox, wcci_test
        environ = {"l2rpn_case14_sandbox": "sandbox", "rte_case14_redisp": "redisp"} # should use regex
        # notes for each agent case and its version
        notes = {
            ("D3QN", 'test'): "In this version\n\
                learning rate 0.95 to 0.85\n\
                Actions: index 0 is DoNothing, reduced actions, setpoint change 0, step 5 for each generator.\n\
                Observations: only rho and prod_p, NO time (gen factor 1.2, flow factor 1.1)"
        }
        # version should change when case is repeated (same env-agent-reward)
        for case_name, version in [("D3QN", 'test')]: #"DoNothing" "ExpertSystem" "D3QN"
            case = "_AGC_{}_{}_{}_v{}".format(environ[env_case_name], case_name, reward_to_use.__name__, version)
            path_save = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases\\{}'.format(case)
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            else:
                shutil.rmtree(path_save)
                os.mkdir(path_save)
            with open(os.path.join(path_save, "NOTES.txt"), "w") as f:
                f.write("{}\n".format(notes[(case_name, version)]))

            start_time = time.time()
            print("-------------{}--------------".format(case_name))
            print("start time: {}".format(start_time))
            env_name = env_case_name
            env = grid2op.make(env_name, reward_class=reward_to_use) 
            # env = grid2op.make(reward_class=L2RPNReward, other_rewards={"other_reward": FlatReward})

            # choose your agent
            if case_name == "DoNothing":
                agent = DoNothingAgent(env.action_space)
            elif case_name == "ExpertSystem":
                tp = TopologyPTDF(env=env, path=path_save)
                agent = ExpertSystem(env.observation_space, env.action_space, tp)
            elif case_name == "D3QN_H":
                tp = TopologyPTDF(env=env, path=path_save)
                agent_name = "{}_ddqn".format(case_name)
                agent_nn_path = os.path.join(path_save, "{}.h5".format(agent_name))
                if not os.path.exists(agent_nn_path):
                    log_path = os.path.join(path_save, "tf_logs_DDDQN")
                    if not os.path.exists(log_path):
                        os.mkdir(log_path)
                    #DEFAULT VALUES: num_frames=4, batch_size=32, lr=1e-5
                    agent = D3QNH(env.observation_space, env.action_space, tph_ptdf=tp, name=agent_name, is_training=True)
                    agent.train(env, train_iter, path_save, logdir=log_path)
                    agent.export_summary(log_path=os.path.join(log_path, agent_name))
                else:
                    agent = D3QNH(env.observation_space, env.action_space, tph_ptdf=tp)
                    agent.load(agent_nn_path)
            elif case_name == "D3QN":
                agent_name = "{}_ddqn".format(case_name)
                agent_nn_path = os.path.join(path_save, "{}.h5".format(agent_name))
                if not os.path.exists(agent_nn_path):
                    log_path = os.path.join(path_save, "tf_logs_DDDQN")
                    if not os.path.exists(log_path):
                        os.mkdir(log_path)
                    #DEFAULT VALUES: num_frames=4, batch_size=32, lr=1e-5
                    agent = D3QN(env.observation_space, env.action_space, name=agent_name, is_training=True)
                    agent.train(env, train_iter, path_save, logdir=log_path)
                    agent.export_summary(log_path=os.path.join(log_path, agent_name))
                else:
                    agent = D3QN(env.observation_space, env.action_space)
                    agent.load(agent_nn_path)
                    
            # fight!
            if  False:
                # create the proper runner
                dict_params = env.get_params_for_runner()
                runner = MyRunner(**dict_params, agentClass=None, agentInstance=agent)

                # now you can run
                res = runner.run(nb_episode=20, path_save=path_save, pbar=True) # path_save=path_save
                #res = runner.run_one_episode(indx=3, path_save=path_save)
                for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
                        msg_tmp += "\t\t - total score: {:.6f}\n".format(cum_reward)
                        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
                        print(msg_tmp)
            
            print("--- {} hours ---".format((time.time() - start_time)/3600))