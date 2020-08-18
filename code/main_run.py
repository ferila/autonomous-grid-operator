import os
import time
import grid2op
from MyRunner import MyRunner
from Heuristic import TopologyPTDF
from grid2op.Reward import L2RPNReward
from grid2op.Agent import DoNothingAgent
from MyAgents.DQNb import MyDoubleDuelingDQN
from MyAgents.FirstAgents import MyFirstPTDFAgent
from l2rpn_baselines.DoubleDuelingDQN import train as DDDQN_train

# tOne: with load
# tOne: with different range

if __name__ == "__main__":
    for env_case_name in ["l2rpn_case14_sandbox"]: # "rte_case14_redisp"
        for case_name in ["MyDDQN"]: #"DoNothing" "MyPTDFAgent" "MyDDQN"
            start_time = time.time()
            print("-------------{}--------------".format(case_name))
            print("start time: {}".format(start_time))

            aa = {"l2rpn_case14_sandbox": "sandbox", "rte_case14_redisp": "redisp"} # should use regex
            case = "tTwo_{}_{}".format(aa[env_case_name], case_name)
            path_save = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases\\{}'.format(case)
            if not os.path.exists(path_save):
                os.mkdir(path_save)

            #TODO from grid2op.Reward import GameplayReward
            #TODO env = grid2op.make("l2rpn_wcci_2020", reward_class=GameplayReward)
            env_name = env_case_name # rte_case14_redisp, l2rpn_case14_sandbox, wcci_test
            env = grid2op.make(env_name, reward_class=L2RPNReward) 
            # from grid2op.Reward import L2RPNReward, FlatReward
            # env = grid2op.make(reward_class=L2RPNReward,
            #                    other_rewards={"other_reward": FlatReward})

            # choose your agent
            if case_name == "DoNothing":
                agent = DoNothingAgent(env.action_space)
            elif case_name == "MyPTDFAgent":
                tp = TopologyPTDF(env=env, path=path_save)
                agent = MyFirstPTDFAgent(env.action_space, tp)
            elif case_name == "MyDDQN_H":
                # TODO
                tp = TopologyPTDF(env=env, path=path_save)
            elif case_name == "MyDDQN":
                agent_name = "{}_ddqn".format(case_name)
                agent_nn_path = os.path.join(path_save, "{}.h5".format(agent_name))
                if not os.path.exists(agent_nn_path):
                    train_iter = 25000
                    log_path = os.path.join(path_save, "tf_logs_DDDQN")
                    if not os.path.exists(log_path):
                        os.mkdir(log_path)
                    #DEFAULT VALUES: num_frames=4, batch_size=32, lr=1e-5
                    agent = MyDoubleDuelingDQN(env.observation_space, env.action_space, name=agent_name, is_training=True)
                    agent.train(env, train_iter, path_save, logdir=log_path)
                    agent.export_summary(log_path=os.path.join(log_path, agent_name))
                else:
                    agent = MyDoubleDuelingDQN(env.observation_space, env.action_space)
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
            
            print("--- {} hours ---".format((time.time() - start_time)/3600))