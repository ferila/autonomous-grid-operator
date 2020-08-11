import time
import numpy as np
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
from grid2op.dtypes import dt_int, dt_float, dt_bool

class MyRunner(Runner):
    """
    It is like the Runner class but the idea is to handle the run parts for my own purposes.
    It initialises from Runner class in the same way.
    """
    @staticmethod
    def _run_one_episode(env, agent, logger, indx, path_save=None, pbar=False, seed=None, max_iter=None):
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
            while not done:
                beg__ = time.time()
                act = agent.act(obs, reward, done)
                end__ = time.time()
                time_act += end__ - beg__

                obs, reward, done, info = env.step(act)  # should load the first time stamp
                cum_reward += reward
                time_step += 1
                pbar_.update(1)

                episode.incr_store(efficient_storing, time_step, end__ - beg__,
                                   float(reward), env.env_modification, act, obs, info)
            end_ = time.time()

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