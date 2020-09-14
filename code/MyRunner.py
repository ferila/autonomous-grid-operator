import os
import time
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from cycler import cycler
import matplotlib.pyplot as plt
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
from grid2op.dtypes import dt_int, dt_float, dt_bool

class MyRunner(Runner):
    """
    It is like the Runner class but the idea is to handle the run parts for my own purposes.
    It initialises from Runner class in the same way.
    """
    

    @staticmethod
    def _run_one_episode(env, agent, logger, indx, path_save=None, pbar=False, max_iter=None, seed=None):
        def _FR_append_observations(dict_obs, act, obs, rew):
            if dict_obs:
                dict_obs['prod_p'].append(obs.prod_p)
                dict_obs['rho'].append(obs.rho)
                dict_obs['actual_dispatch'].append(obs.actual_dispatch)
                dict_obs['pmax'].append(obs.gen_pmax)
                dict_obs['timestep_overflow'].append(obs.timestep_overflow)
                dict_obs['rewards'].append(rew)
                dict_obs['target_dispatch'].append(obs.target_dispatch)
                if act.as_dict():
                    dict_obs['redispatch'].append(act.as_dict()['redispatch'])
                else:
                    dict_obs['redispatch'].append([0., 0., 0., 0., 0., 0.])
            else:
                dict_obs['prod_p'] = [obs.prod_p]
                dict_obs['rho'] = [obs.rho]
                dict_obs['actual_dispatch'] = [obs.actual_dispatch]
                dict_obs['pmax'] = [obs.gen_pmax]
                dict_obs['timestep_overflow'] = [obs.timestep_overflow]
                dict_obs['rewards'] = [rew]
                dict_obs['target_dispatch'] = [obs.target_dispatch]
                if act.as_dict():
                    dict_obs['redispatch'] = [act.as_dict()['redispatch']]
                else:
                    dict_obs['redispatch'] = [[0., 0., 0., 0., 0., 0.]]
                
            return dict_obs

        def _FR_export_dict_ObsActs(path_save, dict_obs, episode):
            def _overflow_redispatch_plot(save_path, dict_obs, episode):
                fontsize = 8
                resolution = 720
                fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 'height_ratios': [2,1]})
                
                a0.set_title("Overflows")
                a0.set_ylabel("Lines", fontsize=fontsize)

                a1.set_title("Redispatch")
                a1.set_ylabel("Power [MW]", fontsize=fontsize)

                plays = len(dict_obs['rho'])
                # actual_d = self._get_all_values(obs, 'actual_dispatch')
                redisp_act = np.array(dict_obs['redispatch'])
                rho = np.array(dict_obs['rho'])
                disp_available = [True, True, False, False, False, True]
                disp = np.array([0,1,5])

                x = np.arange(plays)
                normalize = mpl.colors.Normalize(vmin=0, vmax=1.3)
                for lix in range(rho.shape[1]):
                    a0.scatter(x, ["l{}".format(lix) for i in range(plays)], c=rho[:, lix], cmap=cm.Reds, norm=normalize, alpha=0.5)
                
                colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(disp))]
                a1.set_prop_cycle(cycler('color', colors))
                a1.plot(x, redisp_act[:,disp_available], alpha=0.7)
                a1.legend()

                fig.tight_layout()
                fig.savefig(os.path.join(save_path, "{}_overflow_dispatch".format(episode)), dpi=resolution)
                plt.close(fig)
                
            def _generation_plot(path_save, dict_obs, episode):
                alpha = 0.7
                fontsize = 8
                resolution = 720
                fig2, ax = plt.subplots(1, squeeze=False)
                ax[0,0].set_title("This Case")
                ax[0,0].set_ylabel("Generation [MW]", fontsize=fontsize)
                ax[0,0].set_xlabel("Steps", fontsize=fontsize) 
                
                gen = np.array(dict_obs['prod_p'])
                gmax = np.array(dict_obs['pmax'])

                plays = len(dict_obs['prod_p'])
                x = np.arange(plays)
                
                # show only redispatchables generators
                disp_available = [True, True, False, False, False, True]
                disp = np.array([0,1,5])
                colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(disp))]
                ax[0,0].set_prop_cycle(cycler('color', colors))
                ax[0,0].plot(x, gen[:,disp_available], alpha=alpha)
                ax[0,0].plot(x, gmax[:,disp_available], alpha=alpha)
            
                ax[0,0].legend(disp)

                fig2.tight_layout()
                fig2.savefig(os.path.join(path_save, "{}_generation".format(episode)), dpi=resolution)
                plt.close(fig2)
            
            def _lines_plot(path_save, dict_obs, episode):
                alpha = 0.7
                fontsize = 8
                resolution = 720
                fig2, ax = plt.subplots(2)

                ax[0].set_title("Rewards per timestep")
                ax[0].set_ylabel("Rewards", fontsize=fontsize)
                
                #ax[1].set_title("Disconnected lines")
                #ax[1].set_ylabel("Number of lines", fontsize=fontsize)

                ax[1].set_title('Overflow timesteps')
                ax[1].set_ylabel('Timesteps', fontsize=fontsize)
                ax[1].set_xlabel('Steps', fontsize=fontsize)

                plays = len(dict_obs['prod_p'])
                x = np.arange(plays)
                #disc_lines_accumulated = np.add.accumulate(np.sum(this_episode.disc_lines[0:plays], axis=1))
                overflow_times = np.sum(np.array(dict_obs['timestep_overflow']), axis=1)
                
                ax[0].plot(x, np.array(dict_obs['rewards']), alpha=alpha)
                #ax[1].plot(x, disc_lines_accumulated, label=self.short_names[case_ix], alpha=self.alpha)
                ax[1].plot(x, overflow_times, alpha=alpha)

                ax[0].legend()
                ax[1].legend()
                #ax[2].legend()

                fig2.tight_layout()
                fig2.savefig(os.path.join(path_save, "{}_lines".format(episode)), dpi=resolution)
                plt.close(fig2)

            def _dispatch_plot(path_save, dict_obs, episode):
                alpha = 0.7
                fontsize = 8
                resolution = 720
                fig, ax = plt.subplots(3)

                ax[0].set_title("Actual dispatch")
                ax[0].set_ylabel("Power [MW]", fontsize=fontsize)
                
                ax[1].set_title("Target dispatch")
                ax[1].set_ylabel("Power [MW]", fontsize=fontsize)
                
                ax[2].set_title("Redispatch action")
                ax[2].set_ylabel("Power [MW]", fontsize=fontsize)
                ax[2].set_xlabel("Steps", fontsize=fontsize) 

                actual_d = np.array(dict_obs['actual_dispatch'])
                target_d = np.array(dict_obs['target_dispatch'])
                redisp_act = np.array(dict_obs['redispatch'])

                plays = len(dict_obs['prod_p'])
                x = np.arange(plays)

                # show only redispatchables generators
                disp_available = [True, True, False, False, False, True]
                disp = np.array([0,1,5])
                
                colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(disp))]
                ax[0].set_prop_cycle(cycler('color', colors))
                ax[1].set_prop_cycle(cycler('color', colors))
                ax[2].set_prop_cycle(cycler('color', colors))

                ax[0].plot(x, actual_d[:,disp_available], alpha=alpha)
                ax[1].plot(x, target_d[:,disp_available], alpha=alpha)
                ax[2].plot(x, redisp_act[:,disp_available], alpha=alpha)

                ax[0].legend(disp)
                ax[1].legend(disp)
                ax[2].legend(disp)

                fig.tight_layout()
                fig.savefig(os.path.join(path_save, "{}_dispatches".format(episode)), dpi=resolution)
                plt.close(fig)
            
            path_folder = os.path.join(path_save,"{}_results".format(episode))
            os.mkdir(path_folder)
            _overflow_redispatch_plot(path_folder, dict_obs, episode)
            _generation_plot(path_folder, dict_obs, episode)
            _lines_plot(path_folder, dict_obs, episode)
            _dispatch_plot(path_folder, dict_obs, episode)

        done = False
        time_step = int(0)
        time_act = 0.
        cum_reward = dt_float(0.0)

        # reset the environment
        env.chronics_handler.tell_id(indx-1)
        # the "-1" above is because the environment will be reset. So it will increase id of 1.

        # set the seed
        if seed is not None:
            env.seed(seed)

        # handle max_iter
        if max_iter is not None:
            env.chronics_handler.set_max_iter(max_iter)

        # reset it
        obs = env.reset()

        # reset the agent
        agent.reset(obs)

        # compute the size and everything if it needs to be stored
        nb_timestep_max = env.chronics_handler.max_timestep()
        efficient_storing = nb_timestep_max > 0
        nb_timestep_max = max(nb_timestep_max, 0)

        if path_save is None:
            # i don't store anything on drive, so i don't need to store anything on memory
            nb_timestep_max = 0

        if efficient_storing:
            times = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
            rewards = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
            actions = np.full((nb_timestep_max, env.action_space.n),
                              fill_value=np.NaN, dtype=dt_float)
            env_actions = np.full(
                (nb_timestep_max, env.helper_action_env.n), fill_value=np.NaN, dtype=dt_float)
            observations = np.full(
                (nb_timestep_max+1, env.observation_space.n), fill_value=np.NaN, dtype=dt_float)
            disc_lines = np.full(
                (nb_timestep_max, env.backend.n_line), fill_value=np.NaN, dtype=dt_bool)
            disc_lines_templ = np.full(
                (1, env.backend.n_line), fill_value=False, dtype=dt_bool)
        else:
            times = np.full(0, fill_value=np.NaN, dtype=dt_float)
            rewards = np.full(0, fill_value=np.NaN, dtype=dt_float)
            actions = np.full((0, env.action_space.n), fill_value=np.NaN, dtype=dt_float)
            env_actions = np.full((0, env.helper_action_env.n), fill_value=np.NaN, dtype=dt_float)
            observations = np.full((0, env.observation_space.n), fill_value=np.NaN, dtype=dt_float)
            disc_lines = np.full((0, env.backend.n_line), fill_value=np.NaN, dtype=dt_bool)
            disc_lines_templ = np.full( (1, env.backend.n_line), fill_value=False, dtype=dt_bool)

        if path_save is not None:
            # store observation at timestep 0
            if efficient_storing:
                observations[time_step, :] = obs.to_vect()
            else:
                observations = np.concatenate((observations, obs.to_vect().reshape(1, -1)))

        episode = EpisodeData(actions=actions,
                              env_actions=env_actions,
                              observations=observations,
                              rewards=rewards,
                              disc_lines=disc_lines,
                              times=times,
                              observation_space=env.observation_space,
                              action_space=env.action_space,
                              helper_action_env=env.helper_action_env,
                              path_save=path_save,
                              disc_lines_templ=disc_lines_templ,
                              logger=logger,
                              name=env.chronics_handler.get_name(),
                              other_rewards=[])

        episode.set_parameters(env)

        beg_ = time.time()

        reward = float(env.reward_range[0])
        done = False

        next_pbar = [False]
        with Runner._make_progress_bar(pbar, nb_timestep_max, next_pbar) as pbar_:
            dict_obs = {} #### @felipe
            while not done:
                beg__ = time.time()
                act = agent.act(obs, reward, done)
                end__ = time.time()
                time_act += end__ - beg__

                obs, reward, done, info = env.step(act)  # should load the first time stamp
                if env.parameters.to_dict()['ENV_DC']: #### @felipe
                    dict_obs = _FR_append_observations(dict_obs, act, obs, reward) #### @felipe
                cum_reward += reward
                time_step += 1
                pbar_.update(1)

                episode.incr_store(efficient_storing, time_step, end__ - beg__,
                                   float(reward), env.env_modification, act, obs, info)
            end_ = time.time()
            if env.parameters.to_dict()['ENV_DC']: #### @felipe
                _FR_export_dict_ObsActs(path_save, dict_obs, indx) #### @felipe

            with open(os.path.join(path_save, 'infoEpisodes.txt'), 'a') as f: ##### @felipe
                f.write("--- Episode {} ---\n".format(indx)) ##### @felipe
                f.write("{}\n".format(str(info))) ##### @felipe

        episode.set_meta(env, time_step, float(cum_reward), seed)

        li_text = ["Env: {:.2f}s", "\t - apply act {:.2f}s", "\t - run pf: {:.2f}s",
                   "\t - env update + observation: {:.2f}s", "Agent: {:.2f}s", "Total time: {:.2f}s",
                   "Cumulative reward: {:1f}"]
        msg_ = "\n".join(li_text)
        logger.info(msg_.format(
            env._time_apply_act + env._time_powerflow + env._time_extract_obs,
            env._time_apply_act, env._time_powerflow, env._time_extract_obs,
            time_act, end_ - beg_, cum_reward))

        episode.set_episode_times(env, time_act, beg_, end_)

        episode.to_disk()
        name_chron = env.chronics_handler.get_name()

        return name_chron, cum_reward, int(time_step)