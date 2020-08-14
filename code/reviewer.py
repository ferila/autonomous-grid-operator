import os
import copy
import imageio
import numpy as np
from cycler import cycler
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from grid2op.Episode import EpisodeData
from grid2op.PlotGrid import PlotMatplot

mpl.rcParams['font.size'] = 8
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.titlesize'] = 'small'

class Reviwer(object):
    def __init__(self, path_save, agent_paths, name="default", short_names=[]):
        self.resolution = 720
        self.fontsize = 8
        self.alpha = 0.7
        self.name = name
        self.path_save = path_save
        self.agent_paths = agent_paths
        self.short_names = short_names

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
        width = 0.25
        for ix, c in enumerate(cases):
            x = np.arange(len(res_case[c]['ep']))
            axs[0].bar(x+ix*width, res_case[c]['tot_plays'], width=width, label=self.short_names[ix], align='edge')
            axs[1].bar(x+ix*width, res_case[c]['cum_reward'], width=width, label=self.short_names[ix], align='edge')
        
            for i, p in enumerate(res_case[c]['tot_plays']):
                axs[0].text(i+ix*width, p + 50, str(p), rotation=90, 
                                                                fontsize='xx-small',
                                                                multialignment='left')

            for i, r in enumerate(res_case[c]['cum_reward']):
                axs[1].text(i+ix*width, r + 50, str(int(r)), rotation=90, 
                                                                        fontsize='xx-small',
                                                                        multialignment='left')
        
        [spine.set_visible(False) for spine in axs[0].spines.values()]
        [spine.set_visible(False) for spine in axs[1].spines.values()]
        axs[0].legend()
        axs[1].legend()
        fig.tight_layout()
        #from matplotlib.legend import Legend
        #leg = Legend(ax, lines[2:], ['line C', 'line D'], loc='lower right', frameon=False)
        #ax.add_artist(leg)
        fig.savefig(os.path.join(self.path_save,"{}_episodes_resume".format(self.name)), dpi=self.resolution)


    def analise_episode(self, episode_studied, avoid_agents=[]):
        
        cases = [os.path.join(self.path_save, p) for p in self.agent_paths if p not in avoid_agents]

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
            axs[0].plot(x, this_episode.rewards[0:plays], label=self.short_names[ix], alpha=self.alpha)
            disc_lines_accumulated = np.add.accumulate(np.sum(this_episode.disc_lines[0:plays], axis=1))
            axs[1].plot(x, disc_lines_accumulated, label=self.short_names[ix], alpha=self.alpha)
            
            self._save_grid_images(this_episode, agent_ix=ix, episode_number=episode_studied, last_frames=5)
            self._save_heatmap_images(this_episode, agent_ix=ix, episode_number=episode_studied, last_frames=5)

            axs2[ix].set_title("{}".format(self.short_names[ix]))
            axs2[ix].set_ylabel("Generation [MW]", fontsize=self.fontsize)
            axs2[ix].set_xlabel("Steps", fontsize=self.fontsize) 
            
            obs = copy.deepcopy(this_episode.observations)
            # show only redispatchables generators
            disp_available = obs[-1].gen_redispatchable
            disp = np.arange(obs[-1].n_gen)[disp_available]
            gen = self._get_all_values(obs, 'prod_p')
            gmax = self._get_all_values(obs, 'gen_pmax')
            colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(disp))]
            
            axs2[ix].set_prop_cycle(cycler('color', colors))
            axs2[ix].plot(x, gen[:,disp_available], alpha=self.alpha)
            axs2[ix].plot(x, gmax[:,disp_available], alpha=self.alpha)
            #axs2[ix].plot(x, self._get_all_values(obs, 'gen_pmin'))
        
            axs2[ix].legend(disp)

        axs[0].legend()
        axs[1].legend()
        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig(os.path.join(self.path_save, "{}_rewards_lineDiscon_{}".format(self.name, episode_studied)), dpi=self.resolution)
        fig2.savefig(os.path.join(self.path_save, "{}_generation_{}".format(self.name, episode_studied)), dpi=self.resolution)

    def _get_all_values(self, observations, str_function):
        ans = []
        for i in range(1,len(observations)):
            ans.append(getattr(observations[i], str_function))
        return np.array(ans)

    def _save_heatmap_images(self, this_episode, agent_ix=None, episode_number=None, last_frames=5):
        """For now, only saves an image table of last (frames) values"""
        obs = copy.deepcopy(this_episode.observations)
        rows = obs[-1].n_gen + obs[-1].n_line + obs[-1].n_load
        matrix = np.zeros((rows, last_frames))
        col_names = []
        row_names = []
        for fr in range(last_frames):
            ls = []
            ix = -last_frames + fr
            ls.append(np.round(obs[ix].prod_p))
            row_names.append(obs[ix].name_gen)

            ls.append(np.round(obs[ix].load_p))
            row_names.append(obs[ix].name_load)

            ls.append(np.round(obs[ix].rho, 2))
            row_names.append(obs[ix].name_line)

            matrix[:,fr] = np.concatenate(ls)
            col_names.append("Frame {}".format(ix))

        row_names = np.concatenate(row_names)

        fig, ax = plt.subplots()
        ax.table(cellText = matrix, colLabels = col_names, rowLabels = row_names, loc='center')

        ax.axis('off')
        ax.grid('off')
        image_folder = os.path.join(self.path_save, "{}_{}_{}_images".format(self.name, self.short_names[agent_ix], episode_number))
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        fig.savefig(os.path.join(image_folder, "last_values"), bbox_inches="tight" )

    def _save_grid_images(self, this_episode, agent_ix=None, episode_number=None, last_frames=5):
        """Saves grid with values from the specified last last_frames"""
        images = []
        image_folder = os.path.join(self.path_save, "{}_{}_{}_images".format(self.name, self.short_names[agent_ix], episode_number))
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        obs = copy.deepcopy(this_episode.observations)
        for fr in range(last_frames):
            file_name = "grid_frame{}".format(fr)
            path_name = os.path.join(image_folder, file_name)
            self._plot_grid(this_episode, obs[-last_frames + fr], path=path_name)
            images.append(imageio.imread(path_name))
        
        kwargs = {'duration': 0.8}
        imageio.mimsave(os.path.join(image_folder, "animated_grid"), images, format='GIF', **kwargs)

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

    agent1 = "prstOne_redisp_DoNothing"
    agent2 = "prstOne_redisp_MyDDQN_25000it"
    agent3 = "prstOne_redisp_MyPTDFAgent"
    ag_paths = [agent1, agent2, agent3]
    short_names = ["DoNothing", "D3QN (25k-it)", "ExpertSystem"]

    path_save = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases'
    rev = Reviwer(path_save, ag_paths, name="prstOne_redisp", short_names=short_names)

    rev.resume_episodes() # do a graph for all agents
    
    #rev.analise_episode("0000")
    #rev.analise_episode("0001")
    #rev.analise_episode("0002")


