import os
import time
import grid2op
from MyRunner import MyRunner
from Heuristic import TopologyPTDF
from grid2op.Agent import DoNothingAgent
from MyAgents.DQNb import MyDoubleDuelingDQN
from MyAgents.FirstAgents import MyFirstPTDFAgent
from l2rpn_baselines.DoubleDuelingDQN import train as DDDQN_train


if __name__ == "__main__":
    for case_name in ["MyDDQN"]: #"vDoNothing" "vMyPTDFAgent" "MyDDQN"
        start_time = time.time()
        print("-------------{}--------------".format(case_name))
        print("start time: {}".format(start_time))

        case = "ddqn_tests_sandbox_{}".format(case_name)
        path_save = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases\\{}'.format(case)
        if not os.path.exists(path_save):
            os.mkdir(path_save)

        #TODO from grid2op.Reward import GameplayReward
        #TODO env = grid2op.make("l2rpn_wcci_2020", reward_class=GameplayReward)
        env_name = "l2rpn_case14_sandbox" # rte_case14_redisp, l2rpn_case14_sandbox, wcci_test
        env = grid2op.make(env_name) 
        # from grid2op.Reward import L2RPNReward, FlatReward
        # env = grid2op.make(reward_class=L2RPNReward,
        #                    other_rewards={"other_reward": FlatReward})
        
        # ptdf calculation
        #tp = TopologyPTDF(env=env, path=path_save)
        ###tp = TopologyHeuristic(grid=env.backend._grid, path=path_save)
        ###ptdf = tp.get_ptdf_matrix()
        ###new_lines = tp.get_lines_rewired()

        # choose agent
        if case_name == "vDoNothing":
            agent = DoNothingAgent(env.action_space)
        elif case_name == "vMyPTDFAgent":
            agent = MyFirstPTDFAgent(env.action_space, tp)
        elif case_name == "MyDDQN":
            agent_name = "{}_ddqn".format(case_name)
            agent_nn_path = os.path.join(path_save, "{}.h5".format(agent_name))
            if not os.path.exists(agent_nn_path):
                train_iter = 10000
                log_path = os.path.join(path_save, "tf_logs_DDDQN")
                if not os.path.exists(log_path):
                    os.mkdir(log_path)
                #DEFAULT VALUES: num_frames=4, batch_size=32, lr=1e-5
                agent = MyDoubleDuelingDQN(env.observation_space, env.action_space, name=agent_name, is_training=True)
                #I think next is not necessary
                #if load_path is not None:
                #    agent.load(load_path)
                agent.train(env, train_iter, path_save, logdir=log_path)
            else:
                agent = MyDoubleDuelingDQN(env.observation_space, env.action_space)
                agent.load(agent_nn_path)
                

        if  True:
            # create the proper runner
            dict_params = env.get_params_for_runner()
            runner = MyRunner(**dict_params, agentClass=None, agentInstance=agent)

            # now you can run
            res = runner.run(nb_episode=10, path_save=path_save, pbar=True) # path_save=path_save
            #res = runner.run_one_episode(indx=3, path_save=path_save)
            for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                    msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
                    msg_tmp += "\t\t - total score: {:.6f}\n".format(cum_reward)
                    msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
                    print(msg_tmp)
        
        print("--- {} hours ---".format((time.time() - start_time)/3600))