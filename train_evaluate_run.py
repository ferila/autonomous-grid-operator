import os
import shutil
import grid2op
from grid2op.Runner import Runner
from l2rpn_baselines.DoubleDuelingDQN import DoubleDuelingDQN
from l2rpn_baselines.utils import train_generic

class DoubleDuelingDQN_Improved(DoubleDuelingDQN):    
    def __init__(self, observation_space, action_space, name=__name__, num_frames=4, is_training=False, batch_size=32, lr=1e-5):
        """
        We have changed the size of the observation, so we need to re create another neural network with
        the proper input size. 
        That is why we need to change this.
        """
        # Call parent constructor
        DoubleDuelingDQN.__init__(self,
                                  observation_space=observation_space,
                                  action_space=action_space,
                                  name=name,
                                  num_frames=num_frames,
                                  is_training=is_training,
                                  batch_size=batch_size,
                                  lr=lr)
        
        # import some constant and the class for this baseline
        from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
        from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN import LR_DECAY_STEPS, LR_DECAY_RATE
        
        # Compute dimensions from intial spaces
        self.observation_size = self.obs_space.n_line

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                         self.observation_size,
                                         num_frames = self.num_frames,
                                         learning_rate = self.lr,
                                         learning_rate_decay_steps = LR_DECAY_STEPS,
                                         learning_rate_decay_rate = LR_DECAY_RATE)
        
        # Setup training vars if needed
        if self.is_training:
            self._init_training()
    
    def convert_obs(self, observation):
        """
        And by just changing that, i can change what is fed to the neural network :-)
        
        NB: i need however to tell in the initialization of the neural network the changes I made...
        """
        return observation.rho

class Runna(object):

    def train(env_name, train_iter):
        # create an environment
        env = grid2op.make(env_name)  
        # don't forget to set "test=False" (or remove it, as False is the default value) for "real" training

        # import the train function and train your agent
        agent_name = "test_agent2"
        #save_path = "saved_agent_DDDQN2_{}".format(train_iter)

        my_new_agent = DoubleDuelingDQN_Improved(observation_space=env.observation_space, action_space=env.action_space, is_training=True, name=agent_name)

        my_new_agent_trained = train_generic(agent=my_new_agent,
                                            env=env,
                                            iterations=train_iter,
                                            save_path="saved_agent_DDDQN_{}".format(train_iter))
        
        return my_new_agent_trained

        # DoubleDuelingDQN.train(env, 
        #                         name=agent_name,
        #                         iterations=train_iter,
        #                         save_path=save_path,
        #                         load_path=None, # put something else if you want to reload an agent instead of creating a new one
        #                         logs_path="tf_logs_DDDQN")

    def evaluate(env_name, agent, max_iter):
        ###################
        ### Evaluate Agent
        ###################

        # chose a scoring function (might be different from the reward you use to train your agent)
        #scoring_function = grid2op.L2RPNReward

        # load your agent
        #my_agent = DoubleDuelingDQN_Improved(env.observation_space, env.action_space)
        #my_agent.load(os.path.join(save_path, "{}.h5".format(agent_name)))
        my_agent = agent
        env = grid2op.make(env_name)  

        # here we do that to limit the time take, and will only assess the performance on "max_iter" iteration
        dict_params = env.get_params_for_runner()
        dict_params["gridStateclass_kwargs"]["max_iter"] =  max_iter
        # make a runner from an intialized environment
        runner = Runner(**dict_params, agentClass=None, agentInstance=my_agent)

        return runner

    def run(runner):
        ####################
        ##### Run Agent
        ###################

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

if __name__ == "__main__":
    
    max_iter = 100 # to make computation much faster we will only consider 50 time steps instead of 287
    train_iter = 1000
    env_name = "rte_case14_redisp"

    print("===Training====")
    agent = Runna.train(env_name, train_iter)
    print("===end Training====")
    print("===Evaluating====")
    runner = Runna.evaluate(env_name, agent, max_iter)
    print("===End Evaluating====")
    print("===Running====")
    Runna.run(runner)
    print("===End Running====")


    