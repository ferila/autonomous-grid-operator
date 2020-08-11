import os
import sys
import time
import shutil
import grid2op
import numpy as np
from grid2op.Agent import DoNothingAgent

from MyRunner import Runner
from Heuristic import TopologyPTDF
from MyAgents.DQN.DQNAgent import DQNAgent
from MyAgents.FirstAgents import MyFirstPTDFAgent


if __name__ == "__main__":
    for case_name in ["vMyPTDFAgent"]: #"vDoNothing", 
        start_time = time.time()
        print("start time: {}".format(start_time))

        case = "sandbox_TPH_{}".format(case_name)
        path_save = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\{}'.format(case)
        if not os.path.exists(path_save):
            os.mkdir(path_save)

        #TODO from grid2op.Reward import GameplayReward
        #TODO env = grid2op.make("l2rpn_wcci_2020", reward_class=GameplayReward)
        env_name = "l2rpn_case14_sandbox" # rte_case14_redisp, l2rpn_case14_sandbox, wcci_test
        env = grid2op.make(env_name) 
        
        # ptdf calculation
        tp = TopologyPTDF(env=env, path=path_save)
        #tp = TopologyHeuristic(grid=env.backend._grid, path=path_save)
        #ptdf = tp.get_ptdf_matrix()
        #new_lines = tp.get_lines_rewired()

        # choose agent
        if case_name == "vDoNothing":
            agent = DoNothingAgent(env.action_space)
        elif case_name == "vMyPTDFAgent":
            agent = MyFirstPTDFAgent(env.action_space, tp)
        elif case_name == "DQNNNAgentBLA":
            agent = DQNAgent(env.action_space)
            if _exits_agent_nn(path_save):
                pass
                
            agent.check_training(path_save) #TODO inside: agent.train_generic

        if  True:
            # create the proper runner
            dict_params = env.get_params_for_runner()
            runner = Runner(**dict_params, agentClass=None, agentInstance=agent)

            # now you can run
            res = runner.run(nb_episode=1, path_save=path_save) # path_save=path_save
            #res = runner.run_one_episode(indx=3, path_save=path_save)
            for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                    msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
                    msg_tmp += "\t\t - total score: {:.6f}\n".format(cum_reward)
                    msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
                    print(msg_tmp)
        
        print("--- {} hours ---".format((time.time() - start_time)/3600))

    if False: # notebook version
        # rte_case14_redisp
        # l2rpn_case14_sandbox
        # wcci_test
        from grid2op import make
        env_name = "rte_case14_redisp"
        max_iter = 100
        train_iter = 1000

        """""""""Train agent"""""""""
        # create an environment
        env = make(env_name, test=True)  
        # don't forget to set "test=False" (or remove it, as False is the default value) for "real" training

        # import the train function and train your agent
        from l2rpn_baselines.DoubleDuelingDQN import train
        agent_name = "test_agent"
        save_path = "saved_agent_DDDQN_{}".format(train_iter)
        train(env,
            name=agent_name,
            iterations=train_iter,
            save_path=save_path,
            load_path=None, # put something else if you want to reload an agent instead of creating a new one
            logs_path="tf_logs_DDDQN")

        """""""""Create agent"""""""""
        from grid2op.Runner import Runner

        # chose a scoring function (might be different from the reward you use to train your agent)
        from grid2op.Reward import L2RPNReward
        scoring_function = L2RPNReward

        # load your agent
        from l2rpn_baselines.DoubleDuelingDQN import DoubleDuelingDQN
        my_agent = DoubleDuelingDQN(env.observation_space, env.action_space)
        my_agent.load(os.path.join(save_path, "{}.h5".format(agent_name)))

        # here we do that to limit the time take, and will only assess the performance on "max_iter" iteration
        dict_params = env.get_params_for_runner()
        dict_params["gridStateclass_kwargs"]["max_iter"] =  max_iter
        # make a runner from an intialized environment
        runner = Runner(**dict_params, agentClass=None, agentInstance=my_agent)

        """""""""Run agent and save results"""""""""
        import shutil
        path_save="trained_agent_log"

        # delete the previous stored results
        if os.path.exists(path_save):
            shutil.rmtree(path_save)

        # run the episode
        res = runner.run(nb_episode=2, path_save=path_save)
        print("The results for the trained agent are:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
            msg_tmp += "\t\t - total score: {:.6f}\n".format(cum_reward)
            msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)

        """######################"""
        """Analise agent"""

        """""""""Read from disk and enlist"""""""""
        from grid2op.Episode import EpisodeData
        this_episode = EpisodeData.from_disk(path_save, name="0")
        all_actions = this_episode.get_actions()
        li_actions = []
        for i in range(all_actions.shape[0]):
            try:
                tmp = runner.env.action_space.from_vect(all_actions[i,:])
                li_actions.append(tmp)
            except:
                break
        
        """""""""Which action taken"""""""""
        line_disc = 0
        line_reco = 0
        for act in li_actions:
            dict_ = act.as_dict()
            if "set_line_status" in dict_:
                line_reco +=  dict_["set_line_status"]["nb_connected"]
                line_disc +=  dict_["set_line_status"]["nb_disconnected"]
        print(f'Total reconnected lines : {line_reco}')
        print(f'Total disconnected lines : {line_disc}')

        """""""""Observations recorded"""""""""
        all_observations = this_episode.get_observations()
        li_observations = []
        nb_real_disc = 0
        for i in range(all_observations.shape[0]):
            try:
                tmp = runner.env.observation_space.from_vect(all_observations[i,:])
                li_observations.append(tmp)
                nb_real_disc += (np.sum(tmp.line_status == False))
            except:
                break
        print(f'Total number of disconnected powerlines cumulated over all the timesteps : {nb_real_disc}')

        """""""""Kind of actions selected"""""""""
        actions_count = {}
        for act in li_actions:
            act_as_vect = tuple(act.to_vect())
            if not act_as_vect in actions_count:
                actions_count[act_as_vect] = 0
            actions_count[act_as_vect] += 1
        print("The agent did {} different valid actions:\n".format(len(actions_count)))

        """""""""What actions did"""""""""
        all_act = np.array(list(actions_count.keys()))
        for act in all_act:
            print(runner.env.action_space.from_vect(act))

