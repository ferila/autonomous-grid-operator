import os
import copy
import numpy as np
from cycler import cycler
from matplotlib import cm
import matplotlib.pyplot as plt
from grid2op.Episode import EpisodeData
from grid2op.PlotGrid import PlotMatplot


class Reviwer(object):
    def __init__(self, path_save, agent_paths, name="default"):
        self.resolution = 720
        self.fontsize = 'x-small'
        self.name = name
        self.path_save = path_save
        self.agent_paths = agent_paths

    def resume_episodes(self, avoid_agents=[]):
        """
        Overview of the values for each episode. Per episode, it includes:
        - completed steps
        - reward
        """
        cases = [os.path.join(self.path_save, p) for p in self.agent_paths if p not in avoid_agents]
        res_case = {}
        for c in cases:
            episodes = [ep for ep in os.listdir(c) if os.path.isdir(os.path.join(c, ep)) and not ep.startswith('tf_logs')]
            res_case[c] = {'ep': [],'tot_plays': [], 'max_plays': [], 'cum_reward': []}
            for ep in episodes:
                this_episode = EpisodeData.from_disk(c, ep)
                res_case[c]['ep'].append(ep)
                res_case[c]['tot_plays'].append(this_episode.meta['nb_timestep_played'])
                res_case[c]['max_plays'].append(this_episode.meta['chronics_max_timestep'])
                res_case[c]['cum_reward'].append(this_episode.meta['cumulative_reward'])
                # this_episode.meta['other_rewards']
                # this_episode.episode_times['total']
                #         
        
        fig, axs = plt.subplots(2)
        mplays = res_case[c]['max_plays']
        axs[0].set_title("Total plays (max {})".format(mplays[0])) #this is cheat
        axs[0].set_ylabel('Plays', fontsize=self.fontsize)
        axs[1].set_title("Cumulative Rewards")
        axs[1].set_ylabel('Reward', fontsize=self.fontsize)
        axs[1].set_xlabel('Episode', fontsize=self.fontsize)
        #axs[0].set_xticks(x)
        #axs[0].set_xticklabels(labels)
        #axs[0].set_xticks(x)
        #axs[0].set_xticklabels(labels)
        width = 0.35
        for ix, c in enumerate(cases):
            x = np.arange(len(res_case[c]['ep']))
            axs[0].bar(x+ix*width, res_case[c]['tot_plays'], width, label=self.agent_paths[ix])
            axs[1].bar(x+ix*width, res_case[c]['cum_reward'], width, label=self.agent_paths[ix])
        
            for i, p in enumerate(res_case[c]['tot_plays']):
                axs[0].text(i+ix*width-width/2, p + 50, str(p), rotation=45, 
                                                                fontsize='x-small',
                                                                multialignment='left')

            for i, r in enumerate(res_case[c]['cum_reward']):
                axs[1].text(i+ix*width-width/2, r + 50, str(int(r)), rotation=45, 
                                                                        fontsize='x-small',
                                                                        multialignment='left')
        
        axs[0].legend()
        axs[1].legend()
        fig.tight_layout()
        #from matplotlib.legend import Legend
        #leg = Legend(ax, lines[2:], ['line C', 'line D'], loc='lower right', frameon=False)
        #ax.add_artist(leg)
        fig.savefig(os.path.join(self.path_save,"{}_episodes_resume".format(self.name)), dpi=self.resolution)


    def analise_episode(self, episode_studied, avoid_agents=[]):
        
        cases = [os.path.join(self.path_save, p) for p in self.agent_paths if p not in avoid_agents]
            # rewards, disc_lines, 
            # tot_gen, loads
            # generators
        fig, axs = plt.subplots(2) # Rewards, line disconnections
        fig2, axs2 = plt.subplots(len(cases)) # Production by generator
        
        axs[0].set_title("Rewards per timestep")
        axs[0].set_ylabel("Rewards", fontsize=self.fontsize)
        axs[1].set_title("Disconnected lines")
        axs[1].set_ylabel("Number of lines", fontsize=self.fontsize)
        axs[1].set_xlabel("Steps", fontsize=self.fontsize)
        
        for ix, c in enumerate(cases):
            this_episode = EpisodeData.from_disk(c, episode_studied)
           
            plays = this_episode.meta['nb_timestep_played']
            x = np.arange(plays)
            axs[0].plot(x, this_episode.rewards[0:plays], label=self.agent_paths[ix])
            axs[1].plot(x, np.sum(this_episode.disc_lines[0:plays], axis=1), label=self.agent_paths[ix])
            
            obs = copy.deepcopy(this_episode.observations)
            self._plot_grid(this_episode, obs[-1], path=path_save)

            axs2[ix].set_title("{}".format(self.agent_paths[ix]))
            axs2[ix].set_ylabel("Generation [MW]", fontsize=self.fontsize)
            axs2[ix].set_xlabel("Steps", fontsize=self.fontsize) 
            
            # show only redispatchables generators
            disp_available = obs[-1].gen_redispatchable
            disp = np.arange(obs[-1].n_gen)[disp_available]
            gen = self._get_all_values(obs, 'prod_p')
            gmax = self._get_all_values(obs, 'gen_pmax')
            colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(disp))]
            
            axs2[ix].set_prop_cycle(cycler('color', colors))
            axs2[ix].plot(x, gen[:,disp_available])#, color=colors[i])
            axs2[ix].plot(x, gmax[:,disp_available])#, color=colors[i])
            #axs2[ix].plot(x, self._get_all_values(obs, 'gen_pmin'))
        
        axs[0].legend()
        axs[1].legend()
        axs2[0].legend(disp)
        axs2[1].legend(disp)
        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig(os.path.join(self.path_save, "{}_rewards_lineDiscon".format(self.name)), dpi=self.resolution)
        fig2.savefig(os.path.join(self.path_save, "{}_generation".format(self.name)), dpi=self.resolution)

    def _get_all_values(self, observations, str_function):
        ans = []
        for i in range(1,len(observations)):
            ans.append(getattr(observations[i], str_function))
        return np.array(ans)

    def _plot_grid(self, this_episode, obs, show=False, path=None):
        plot_helper = PlotMatplot(observation_space=this_episode.observation_space, width=1920, height=1080)
        plot_helper._line_bus_radius = 7
        fig = plot_helper.plot_obs(obs)
        if show:
            fig.show()
        if not path is None:
            fig.savefig(path)
        plt.close(fig)

if __name__ == "__main__":

    agent1 = "ddqn_tests_sandbox_vDoNothing"
    agent2 = "ddqn_tests_sandbox_MyDDQN_10000it"
    ag_paths = [agent1, agent2]

    path_save = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases'
    rev = Reviwer(path_save, ag_paths, name="analysis1")

    rev.resume_episodes() # do a graph for all agents
    
    rev.analise_episode("0000")


