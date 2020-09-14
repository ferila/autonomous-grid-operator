import os
import time
import shutil
import grid2op
from MyRunner import MyRunner
from MyReviewer import Reviewer
from Heuristic import TopologyPTDF
from grid2op.Reward import L2RPNReward, RedispReward
from MyRewards.XDiReward import Di24Reward, CDi24Reward, CDi242Reward # LidiReward, OvdiReward, DiReward, FoolReward, Di2Reward, Di3Reward, CDi3Reward, C2Di3Reward, Di22Reward, C3Di3Reward, Di23Reward, 
from grid2op.Agent import DoNothingAgent
from MyAgents.DQNb import D3QN
from MyAgents.ExpertSystem import ExpertSystem
from grid2op.Parameters import Parameters
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
    #reward_to_use = DiReward #GameplayReward in "l2rpn_wcci_2020"
    train_iter = 35000
    NUM_FRAMES = 3
    controllable_generators = [5] # must be sorted increasingly
    custom_params = Parameters()
    custom_params.ENV_DC = False
    for env_case_name in ["l2rpn_case14_sandbox"]: # "rte_case14_redisp" # rte_case14_redisp, l2rpn_case14_sandbox, wcci_test
        environ = {"l2rpn_case14_sandbox": "sandbox", "rte_case14_redisp": "redisp"} # should use regex
        # notes for each agent case and its version
        vnotes = {
            'c2di3': "In this version:\n\
                Reward: continuous values, no action cost, no is_illegal, min_rew (-20), ovf threshold 0.5, safety_redisp 0.95, overflow (-20), target_dispatch distance (-10).\n\
                Hyperparams: num_frames 3, 35k it. \n\
                Actions: RedispatchActionFactory. control_gen=[5].\n\
                Observations: rho, prod_p and time (min,hour,day) (gen norm factor 1.2, flow norm factor 1.1)",
            'di2': "In this version:\n\
                Reward: continuous values, no action cost, no is_illegal, min_rew (-10), ovf threshold 0.95, overflow (-20), generation limits (-10).\n\
                Hyperparams: num_frames 3, 35k it. \n\
                Actions: RedispatchActionFactory. control_gen=[5].\n\
                Observations: rho, prod_p and time (min,hour,day) (gen norm factor 1.2, flow norm factor 1.1)",
            'DC_pq_ctrl5': "In this version:\n\
                env: DC mode. \n\
                Reward: discrete values, no action cost, no is_illegal, min_rew (-10), ovf threshold 0.5, overflow (-20), generation limits (-10).\n\
                Hyperparams: num_frames 3, 35k it. \n\
                Actions: controlling generator 5.\n\
                Observations: rho, prod_p, prod_q and time (min,hour,day) (p and q normalised with apparent power)",
            'final_single_seed1': "In this version:\n\
                env: AC mode. \n\
                Reward: ovf (-40), actual_disp (-30), no diff dispatch (-0), is illegal (-10). ovf threshold 0. Rew min -100 \n\
                Hyperparams: num_frames 3, 35k it. \n\
                Actions: controlling all generator. Ramps values [(-2.5,2.5), (-5,5), (-5,5)]. ramp step 2.5. \n\
                Observations: rho, load_p, prod_p, pfactor (prod_p, prod_q) and time (min,hour,day) (p and q normalised with apparent power)",
            'L2RPN': "In this version:\n\
                env: AC mode. \n\
                Hyperparams: num_frames 3, 35k it. \n\
                Actions: controlling all generator. Ramps values [(-2.5,2.5), (-5,5), (-5,5)]. ramp step 2.5. \n\
                Observations: rho, prod_p, pfactor (prod_p, prod_q) and time (min,hour,day) (p and q normalised with apparent power)",
            'AC_ExpSys_ovf.3': "overflow threshold 0.3"
        }
        # version should change when case is repeated (same env-agent-reward)
        for case_name, version, reward_to_use in [("D3QN", 'final_single_seed1', CDi242Reward)]: #, ("D3QN", 'c2di3', C2Di3Reward)]: #"DoNothing" "ExpertSystem" "D3QN"
            case = "_AGC_{}_{}_{}_({})".format(environ[env_case_name], case_name, reward_to_use.__name__, version)
            path_save = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases\\{}'.format(case)
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            else:
                raise Exception('Folder already exists')
            with open(os.path.join(path_save, "NOTES.txt"), "w") as f:
                f.write("{}\n".format(vnotes[version]))

            start_time = time.time()
            print("-------------{}--------------".format(case_name))
            print("start time: {}".format(start_time))
            env_name = env_case_name
            env = grid2op.make(env_name, reward_class=reward_to_use, param=custom_params)
            print(env.parameters.to_dict())
            # env = grid2op.make(reward_class=L2RPNReward, other_rewards={"other_reward": FlatReward})

            # choose your agent
            if case_name == "DoNothing":
                agent = DoNothingAgent(env.action_space)
            elif case_name == "ExpertSystem":
                tp = TopologyPTDF(env=env, path=path_save)
                agent = ExpertSystem(env.observation_space, env.action_space, tp)
            elif case_name == "D3QN_H":
                pass
            elif case_name == "D3QN":
                agent_name = "{}_ddqn".format(case_name)
                agent_nn_path = os.path.join(path_save, "{}.h5".format(agent_name))
                if not os.path.exists(agent_nn_path):
                    log_path = os.path.join(path_save, "tf_logs_DDDQN")
                    if not os.path.exists(log_path):
                        os.mkdir(log_path)
                    #DEFAULT VALUES: num_frames=4, batch_size=32, lr=1e-5
                    agent = D3QN(env.observation_space, env.action_space, name=agent_name, is_training=True, num_frames=NUM_FRAMES, control_gen=controllable_generators)
                    agent.train(env, train_iter, path_save, logdir=log_path)
                    agent.export_summary(log_path=os.path.join(log_path, agent_name))
                else:
                    agent = D3QN(env.observation_space, env.action_space, num_frames=NUM_FRAMES)
                    agent.load(agent_nn_path)
                    
            # fight!
            if  True:
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
                

                # exporting analysis
                analysis_folder_name = "agent_analysis"
                ag_paths = [case]
                short_names = [case_name]
                notes = [vnotes[version]]
                selected_episodes = ["0002", "0003", "0007", "0011", "0012", "0017", "0018"]

                run_path = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases'
                analysis_path = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases\\{}\\_STDY_{}'.format(case, analysis_folder_name)
                rev = Reviewer(run_path, analysis_path, ag_paths, name='a', short_names=short_names, notes=notes, self_analysis=True)

                rev.resume_episodes()
                for ep in selected_episodes:
                    rev.analise_episode(ep)
            
            print("--- {} hours ---".format((time.time() - start_time)/3600))