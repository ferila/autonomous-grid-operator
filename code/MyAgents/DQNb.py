#from DQN_NNb import DQN_NNb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .Actions import RedispatchActions, RedispatchSpecificActions, RedispatchActionFactory
from l2rpn_baselines.DoubleDuelingDQN import DoubleDuelingDQN
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class D3QN(DoubleDuelingDQN):

    def __init__(self, observation_space, action_space, name=__name__, is_training=False, num_frames=4, batch_size=32, lr=1e-5, control_gen=[]):
        super().__init__(observation_space,
                        action_space,
                        name=name,
                        is_training=is_training)
        
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.lr = lr

        red_acts = RedispatchSpecificActions(observation_space, action_space, max_setpoint_change=0)
        self.redispatch_actions_dict = red_acts.REDISPATCH_ACTIONS_DICT
        #red_acts = RedispatchActionFactory(observation_space, controllable_generators=control_gen, maximum_ramp_gap=2)
        #self.redispatch_actions_dict = red_acts.redispatch_actions_dict
        print(self.redispatch_actions_dict)
        # v1: observation size = powerflows + generators (prod_p) + (minute, hour, day and month)
        # v2: observation size = powerflows + loads (load_p) + generators (prod_p) + (minute, hour, day and month)
        # v3: observation size = powerflows + generators (prod_p) + generators (actual_d) + (minute, hour, day and month)
        self.observation_size = self.obs_space.n_line + self.obs_space.n_gen + 3
        # action size = decrease (max_ramp_down), stay or increase (max_ramp_up) dispatch for each redispatchable generator
        self.action_size = len(self.redispatch_actions_dict) #self.ACTIONS_PER_GEN ** sum(self.obs_space.gen_redispatchable)

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                        self.observation_size,
                                        num_frames = self.num_frames,
                                        learning_rate = self.lr,
                                        learning_rate_decay_steps = cfg.LR_DECAY_STEPS,
                                        learning_rate_decay_rate = cfg.LR_DECAY_RATE)
        
        # Setup training vars if needed
        if self.is_training:
            self._init_training()
        
    def convert_obs(self, observation):
        res = []

        if True:
            # include powerflow observations
            powerflow_limit = 1.1 # 200% of powerflow disconnects automatically the powerline
            max_val = powerflow_limit
            min_val = -powerflow_limit
            # new_val_x = min_new + (max_new - min_new) * (val_x - min_x) / (max_x - min_x)
            #pflow = -1 + 2 * (np.sign(observation.p_or) * observation.rho - min_val) / (max_val - min_val) 
            pflow = (np.sign(observation.p_or) * abs(observation.rho) - min_val) / (max_val - min_val) 
            res.append(pflow)
        
        if False:
            # include load_p observations
            max_sorted = np.sort(observation.gen_pmax)
            max_load = max_sorted[-1] + max_sorted[-2] # normalise by the two biggest generators
            load_norm = observation.load_p / max_load
            res.append(load_norm)
        
        if True:
            # include generation observations
            gen = abs(observation.prod_p) / (observation.gen_pmax * 1.2) # 1.1 (bef 1.1) because prod_p seems to go beyond the maximum
            res.append(gen)
        
        if False:
            # include actual dispatch
            max_disp = observation.gen_pmax
            min_disp = -observation.gen_pmax
            actual_d = (observation.actual_dispatch - min_disp) / (max_disp - min_disp) #0.01 + (1 - 0.01) *
            res.append(actual_d[observation.gen_redispatchable])

        if True:
            # include time (year not included, should work only on long term)
            res.append(np.array([observation.minute_of_hour / 60]))
            res.append(np.array([observation.hour_of_day / 23]))
            res.append(np.array([observation.day_of_week / 7]))
            #res.append(observation.month / 12)
        
        res = np.concatenate(res)
        return res

    def convert_act(self, action):
        """
        Int action from my_act.
        Return a valid action
        """
        # nummber of actions: number of redispatches
        # 0: [(red_gen_id1, 0),(red_gen_id2, 0),(red_gen_id3, 0)]
        return self.action_space({"redispatch": self.redispatch_actions_dict[action]})
    
    def my_act(self, state, reward, done=False):
        # Register current state to stacking buffer
        self._save_current_frame(state)
        # We need at least num frames to predict
        if len(self.frames) < self.num_frames:
            action_selected = 0 # Do nothing
        # Infer with the last num_frames states
        else:
            action_selected, _ = self.Qmain.predict_move(np.array(self.frames))
        return action_selected
    
    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              num_pre_training_steps=0,
              logdir = "logs-train"):
        # Make sure we can fill the experience buffer
        if num_pre_training_steps < self.batch_size * self.num_frames:
            num_pre_training_steps = self.batch_size * self.num_frames

        # Loop vars
        num_training_steps = iterations
        num_steps = num_pre_training_steps + num_training_steps
        step = 0
        self.epsilon = cfg.INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        self.done = True

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".h5")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, num_steps)

        action_backup = np.zeros(num_steps) #### @felipe
        qaction_backup = np.zeros(num_steps) #### @felipe
        qaction_backup[:] = np.NaN #### @felipe
        qval_backup = np.zeros((num_steps, self.action_size)) #### @felipe
        qval_backup[:] = np.nan #### @felipe
        reward_backup = np.zeros(num_steps) #### @felipe
        step_lasted = [] #### @felipe
        loss_backup = np.zeros(num_steps) #### @felipe
        loss_backup[:] = np.nan #### @felipe
        lr_backup = np.zeros(num_steps) #### @felipe
        lr_backup[:] = np.nan #### @felipe

        # Training loop
        while step < num_steps:
            # Init first time or new episode
            if self.done:
                new_obs = env.reset() # This shouldn't raise
                self.reset(new_obs)
            if cfg.VERBOSE and step % 1000 == 0:
                print("Step [{}] -- Random [{}]".format(step, self.epsilon))

            # Save current observation to stacking buffer
            self._save_current_frame(self.state)

            # Choose an action
            if step <= num_pre_training_steps:
                a = self.Qmain.random_move()
            elif np.random.rand(1) < self.epsilon:
                a = self.Qmain.random_move()
            elif len(self.frames) < self.num_frames:
                a = 0 # Do nothing
            else:
                a, q_actions = self.Qmain.predict_move(np.array(self.frames))
                qval_backup[step] = q_actions #### @felipe
                qaction_backup[step] = a #### @felipe
            action_backup[step] = a #### @felipe


            # Convert it to a valid action
            act = self.convert_act(a)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            reward_backup[step] = reward #### @felipe
            new_state = self.convert_obs(new_obs)
            if info["is_illegal"] or info["is_ambiguous"] or \
               info["is_dispatching_illegal"] or info["is_illegal_reco"]:
               pass ###############
                #if cfg.VERBOSE: ############################
                #    print (a, info) ###########################

            # Save new observation to stacking buffer
            self._save_next_frame(new_state)

            # Save to experience buffer
            if len(self.frames2) == self.num_frames:
                self.per_buffer.add(np.array(self.frames),
                                    a, reward,
                                    np.array(self.frames2),
                                    self.done)

            # Perform training when we have enough experience in buffer
            if step >= num_pre_training_steps:
                training_step = step - num_pre_training_steps
                # Decay chance of random action
                self.epsilon = self._adaptive_epsilon_decay(training_step)

                # Perform training at given frequency
                if step % cfg.UPDATE_FREQ == 0 and \
                   len(self.per_buffer) >= self.batch_size:
                    # Perform training
                    loss, lrb = self._batch_train(training_step, step) #### @felipe
                    loss_backup[step] = loss #### @felipe
                    lr_backup[step] = lrb #### @felipe

                    if cfg.UPDATE_TARGET_SOFT_TAU > 0.0:
                        tau = cfg.UPDATE_TARGET_SOFT_TAU
                        # Update target network towards primary network
                        self.Qmain.update_target_soft(self.Qtarget.model, tau)

                # Every UPDATE_TARGET_HARD_FREQ trainings, update target completely
                if cfg.UPDATE_TARGET_HARD_FREQ > 0 and \
                   step % (cfg.UPDATE_FREQ * cfg.UPDATE_TARGET_HARD_FREQ) == 0:
                    self.Qmain.update_target_hard(self.Qtarget.model)

            total_reward += reward
            if self.done:
                self.epoch_rewards.append(total_reward)
                self.epoch_alive.append(alive_steps)
                step_lasted.append(alive_steps) #### @felipe
                if cfg.VERBOSE:
                    print("Survived [{}] steps".format(alive_steps))
                    print("Total reward [{}]".format(total_reward))
                alive_steps = 0
                total_reward = 0
            else:
                alive_steps += 1
            
            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                self.save(modelpath)

            # Iterate to next loop
            step += 1
            # Make new obs the current obs
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.save(modelpath)
        self._save_act_and_qval(loss_backup, lr_backup, action_backup, qaction_backup, qval_backup, reward_backup, step_lasted, path_save=logdir) #### @felipe
    
    def _batch_train(self, training_step, step):
        """Trains network to fit given parameters"""

        # Sample from experience buffer
        sample_batch = self.per_buffer.sample(self.batch_size, cfg.PER_BETA)
        s_batch = sample_batch[0]
        a_batch = sample_batch[1]
        r_batch = sample_batch[2]
        s2_batch = sample_batch[3]
        d_batch = sample_batch[4]
        w_batch = sample_batch[5]
        idx_batch = sample_batch[6]

        Q = np.zeros((self.batch_size, self.action_size))

        # Reshape frames to 1D
        input_size = self.observation_size * self.num_frames
        input_t = np.reshape(s_batch, (self.batch_size, input_size))
        input_t_1 = np.reshape(s2_batch, (self.batch_size, input_size))

        # Save the graph just the first time
        if training_step == 0:
            tf.summary.trace_on()

        # T Batch predict
        Q = self.Qmain.model.predict(input_t, batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if training_step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1 = self.Qmain.model.predict(input_t_1, batch_size=self.batch_size)
        Q2 = self.Qtarget.model.predict(input_t_1, batch_size=self.batch_size)

        # Compute batch Qtarget using Double DQN
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q1[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += cfg.DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.Qmain.train_on_batch(input_t, Q, w_batch)

        # Update PER buffer
        priorities = self.Qmain.batch_sq_error
        # Can't be zero, no upper limit
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        self.per_buffer.update_priorities(idx_batch, priorities)

        # Log some useful metrics every even updates
        if step % (cfg.UPDATE_FREQ * 2) == 0:
            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                if len(self.epoch_rewards) >= 100:
                    mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                    mean_alive_100 = np.mean(self.epoch_alive[-100:])
                else:
                    mean_reward_100 = mean_reward
                    mean_alive_100 = mean_alive
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_100", mean_reward_100, step)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step)
                tf.summary.scalar("loss", loss, step)
                tf.summary.scalar("lr", self.Qmain.train_lr, step)
            if cfg.VERBOSE:
                print("loss =", loss)
        
        return loss, self.Qmain.train_lr

    def _save_act_and_qval(self, loss, lr, actions, qactions, qvalues, rewards, step_lasted, path_save=None):
        loss_df = pd.DataFrame(loss, columns=['loss'])
        loss_df = loss_df.fillna(method='ffill')
        lr_df = pd.DataFrame(lr, columns=['lr'])
        lr_df = lr_df.fillna(method='ffill')
        acts_df = pd.DataFrame(actions, columns=['action'])
        qacts_df = pd.DataFrame(qactions, columns=['action'])
        qvals_df = pd.DataFrame(qvalues, columns=['Qv{}'.format(i) for i in range(qvalues.shape[1])])
        rew_df = pd.DataFrame(rewards, columns=['reward'])
        step_df = pd.DataFrame(step_lasted, columns=['steps'])

        loss_df.to_csv(os.path.join(path_save, "loss.csv"))
        lr_df.to_csv(os.path.join(path_save, "learning_rate.csv"))
        acts_df.to_csv(os.path.join(path_save, "actions.csv"))
        qacts_df.to_csv(os.path.join(path_save, "qactions.csv"))
        qvals_df.to_csv(os.path.join(path_save, "qvalues.csv"))
        rew_df.to_csv(os.path.join(path_save, "rewards.csv"))
        step_df.to_csv(os.path.join(path_save, "steps.csv"))

        # loss and lr
        ax = loss_df.plot()
        fig = ax.get_figure()
        fig.savefig(os.path.join(path_save, '_loss_evolution'), dpi=360)
        plt.close(fig)
        ax = lr_df.plot()
        fig = ax.get_figure()
        fig.savefig(os.path.join(path_save, '_lr_evolution'), dpi=360)
        plt.close(fig)

        # all actions histogram
        ax = acts_df.plot(kind='hist', bins=self.action_size) 
        fig = ax.get_figure()
        fig.savefig(os.path.join(path_save, '_actions_histogram'))
        plt.close(fig)

        # all actions evolution
        acts_df['index'] = acts_df.index
        ax1 = acts_df.plot(kind='scatter', x='index', y='action', alpha=0.05, edgecolors='none')
        fig1 = ax1.get_figure()
        fig1.savefig(os.path.join(path_save, '_action_selection_evolution'), dpi=360)
        plt.close(fig1)

        # q-actions histogram
        ax = qacts_df.plot(kind='hist', bins=self.action_size) 
        fig = ax.get_figure()
        fig.savefig(os.path.join(path_save, '_qactions_histogram'))
        plt.close(fig)

        # q-actions evolution
        qacts_df['index'] = qacts_df.index
        ax1 = qacts_df.plot(kind='scatter', x='index', y='action', alpha=0.05, edgecolors='none')
        fig1 = ax1.get_figure()
        fig1.savefig(os.path.join(path_save, '_qaction_selection_evolution'), dpi=360)
        plt.close(fig1)

        # q-values evolution
        ax2 = qvals_df.plot()
        fig2 = ax2.get_figure()
        fig2.savefig(os.path.join(path_save, '_qvalues_evolution'), dpi=360)
        plt.close(fig2)

        # reward evolution
        ax3 = rew_df.plot()
        fig3 = ax3.get_figure()
        fig3.savefig(os.path.join(path_save, '_reward_evolution'), dpi=360)   
        plt.close(fig3)

        # max steps reached evolution
        ax4 = step_df.plot()
        fig4 = ax4.get_figure()
        fig4.savefig(os.path.join(path_save, '_max_steps_evolution'), dpi=360)      
        plt.close(fig4)

    def export_summary(self, log_path):
        file_path = self._find_file_in(log_path)
        event_acc = EventAccumulator(file_path)
        event_acc.Reload()
        cols = ['wtime', 'step', 'reward', 'alive', 'reward100', 'alive100', 'loss', 'lr']
        res = pd.DataFrame([])
        for metric in ['mean_reward', 'mean_alive', 'mean_reward_100', 'mean_alive_100', 'loss', 'lr']:
            if res.empty:
                res = pd.DataFrame([(w, s, tf.make_ndarray(t)) for w, s, t in event_acc.Tensors(metric)])
            else:
                res  = pd.concat([res, pd.DataFrame([(tf.make_ndarray(t)) for w, s, t in event_acc.Tensors(metric)])], axis=1)
        res.columns = cols
        res.to_csv(os.path.join(log_path, 'train_summary.csv'))
    
    def _find_file_in(self, path):
        files = os.listdir(path)
        fname = files[0] # there is only one file
        return os.path.join(path, fname)
