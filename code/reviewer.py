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

class Reviewer(object):
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
        fig.close()

    def analise_episode(self, episode_studied, last_frames=5, avoid_agents=[]):
        
        cases = [os.path.join(self.path_save, p) for p in self.agent_paths if p not in avoid_agents]

        fig, axs = plt.subplots(2) # Rewards, line disconnections
        fig2, axs2 = plt.subplots(len(cases), squeeze=False) # Production by generator and total load
                
        for ix, c in enumerate(cases):
            this_episode = EpisodeData.from_disk(c, episode_studied)
            image_folder = os.path.join(self.path_save, "{}_{}_images".format(self.agent_paths[ix], episode_studied))
            if not os.path.exists(image_folder):
                os.mkdir(image_folder)

            # Common graphs
            ## Rewards and disconnected lines
            self._add_plot_rewards_and_disconnected_lines(this_episode, axs, case_ix=ix)
            ## Generation of dispatchable generators and net load
            self._add_plot_generations_and_load(this_episode, axs2, subplot_ix=ix)

            # Case specific graphs
            ## Last frames grid images and animated gif
            self._save_grid_images(this_episode, last_frames=last_frames, folder=image_folder)
            ## Last grames values TODO: colour heatmap
            self._save_heatmap_images(this_episode, last_frames=last_frames, folder=image_folder)
            ## Actual dispatch, target dispatch, redispatch actions
            self._add_plot_dispatch_and_redispatch(this_episode, case_ix=ix, folder=image_folder)
            ## Total demand and generation, generation by unit
            self._add_plot_all_generation_by_case(this_episode, case_ix=ix, folder=image_folder)

        # Save common graphs
        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig(os.path.join(self.path_save, "{}_rewards_lineDiscon_{}".format(self.name, episode_studied)), dpi=self.resolution)
        fig2.savefig(os.path.join(self.path_save, "{}_generation_{}".format(self.name, episode_studied)), dpi=self.resolution)
        plt.close(fig)
        plt.close(fig2)

    
    def _add_plot_rewards_and_disconnected_lines(self, this_episode, ax, case_ix=None):
        ax[0].set_title("Rewards per timestep")
        ax[0].set_ylabel("Rewards", fontsize=self.fontsize)
        
        ax[1].set_title("Disconnected lines")
        ax[1].set_ylabel("Number of lines", fontsize=self.fontsize)
        ax[1].set_xlabel("Steps", fontsize=self.fontsize)

        plays = this_episode.meta['nb_timestep_played']
        x = np.arange(plays)
        disc_lines_accumulated = np.add.accumulate(np.sum(this_episode.disc_lines[0:plays], axis=1))
        
        ax[0].plot(x, this_episode.rewards[0:plays], label=self.short_names[case_ix], alpha=self.alpha)
        ax[1].plot(x, disc_lines_accumulated, label=self.short_names[case_ix], alpha=self.alpha)

        ax[0].legend()
        ax[1].legend()
        
    def _add_plot_generations_and_load(self, this_episode, ax, subplot_ix=None):
        ax[subplot_ix,0].set_title("{}".format(self.short_names[subplot_ix]))
        ax[subplot_ix,0].set_ylabel("Generation [MW]", fontsize=self.fontsize)
        ax[subplot_ix,0].set_xlabel("Steps", fontsize=self.fontsize) 
        
        obs = copy.deepcopy(this_episode.observations)
        gen = self._get_all_values(obs, 'prod_p')
        gmax = self._get_all_values(obs, 'gen_pmax')
        loadp = np.sum(self._get_all_values(obs, 'load_p'), axis=1)

        plays = this_episode.meta['nb_timestep_played']
        x = np.arange(plays)
        
        # show only redispatchables generators
        disp_available = obs[-1].gen_redispatchable
        disp = np.arange(obs[-1].n_gen)[disp_available]
        colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(disp))]
        ax[subplot_ix,0].set_prop_cycle(cycler('color', colors))
        ax[subplot_ix,0].plot(x, gen[:,disp_available], alpha=self.alpha)
        ax[subplot_ix,0].plot(x, gmax[:,disp_available], alpha=self.alpha)
        ax[subplot_ix,0].plot(x, loadp, alpha=self.alpha, c='black')
    
        ax[subplot_ix,0].legend(disp)

    
    def _add_plot_dispatch_and_redispatch(self, this_episode, case_ix=None, folder=None):
        """
        # Actual dispatch, target dispatch, redispatch actions
        """
        fig, ax = plt.subplots(3)

        ax[0].set_title("Actual dispatch - {}".format(self.short_names[case_ix]))
        ax[0].set_ylabel("Power [MW]", fontsize=self.fontsize)
        
        ax[1].set_title("Target dispatch - {}".format(self.short_names[case_ix]))
        ax[1].set_ylabel("Power [MW]", fontsize=self.fontsize)
        
        ax[2].set_title("Redispatch action - {}".format(self.short_names[case_ix]))
        ax[2].set_ylabel("Power [MW]", fontsize=self.fontsize)
        ax[2].set_xlabel("Steps", fontsize=self.fontsize) 

        obs = copy.deepcopy(this_episode.observations)
        acts = copy.deepcopy(this_episode.actions)
        actual_d = self._get_all_values(obs, 'actual_dispatch')
        target_d = self._get_all_values(obs, 'target_dispatch')
        redisp_act = self._get_all_action_values(acts, 'redispatch')

        plays = this_episode.meta['nb_timestep_played']
        x = np.arange(plays)

        # show only redispatchables generators
        disp_available = obs[-1].gen_redispatchable
        disp = np.arange(obs[-1].n_gen)[disp_available]
        
        colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(disp))]
        ax[0].set_prop_cycle(cycler('color', colors))
        ax[1].set_prop_cycle(cycler('color', colors))
        ax[2].set_prop_cycle(cycler('color', colors))

        ax[0].plot(x, actual_d[:,disp_available], alpha=self.alpha)
        ax[1].plot(x, target_d[:,disp_available], alpha=self.alpha)
        ax[2].plot(x, redisp_act[:,disp_available], alpha=self.alpha)

        ax[0].legend(disp)
        ax[1].legend(disp)
        ax[2].legend(disp)

        fig.tight_layout()
        fig.savefig(os.path.join(folder, "dispatches"), dpi=self.resolution)
        plt.close(fig)

    def _add_plot_all_generation_by_case(self, this_episode, case_ix=None, folder=None):
        """
        # Total demand and generation, generation by unit
        """
        obs = copy.deepcopy(this_episode.observations)
        n_gen = obs[-1].n_gen
        n_steps = this_episode.meta['nb_timestep_played']
        x = np.arange(n_steps)
        fig, axs = plt.subplots(n_gen + 1, gridspec_kw = {'wspace':0, 'hspace':0})

        load = self._get_all_values(obs, 'load_p')
        gen = self._get_all_values(obs, 'prod_p')
        pmax = self._get_all_values(obs, 'gen_pmax')

        net_load = np.sum(load, axis=1)
        net_gen = np.sum(gen, axis=1)
        axs[0].plot(x, net_load, label='Net load')
        axs[0].plot(x, net_gen, label='Net gen.')
        axs[0].set_xticklabels([])
        axs[0].legend()
        
        colors = [cm.rainbow(x) for x in np.linspace(0, 1, n_gen)]
        for g_ix in range(n_gen):
            axs[g_ix+1].plot(x, gen[:,g_ix], alpha=self.alpha, c=colors[g_ix], label=g_ix)
            axs[g_ix+1].plot(x, pmax[:,g_ix], alpha=self.alpha, c=colors[g_ix])
            axs[g_ix+1].legend()
            if g_ix < n_gen-1:
                axs[g_ix+1].set_xticklabels([])

        #fig.text(-0.5, 0.5, 'Power [MW]', va='center', rotation='vertical')
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "all_generation"), dpi=self.resolution)
        plt.close(fig)

    def _save_heatmap_images(self, this_episode, last_frames=5, folder=None):
        """
        For now, only saves an image table of last (frames) values
        """
        obs = copy.deepcopy(this_episode.observations)
        rows = obs[-1].n_gen + obs[-1].n_line + obs[-1].n_load
        matrix = np.zeros((rows, last_frames))
        col_names = []
        row_names = []
        for fr in range(last_frames):
            ls = []
            ix = -last_frames + fr

            ls.append(np.round(obs[ix].prod_p))
            ls.append(np.round(obs[ix].load_p))
            ls.append(np.round(obs[ix].rho, 2))

            matrix[:,fr] = np.round(np.concatenate(ls), 2)
            col_names.append("Frame {}".format(ix))

        row_names.append(obs[ix].name_gen)
        row_names.append(obs[ix].name_load)
        row_names.append(obs[ix].name_line)
        row_names = np.concatenate(row_names)

        fig, ax = plt.subplots()
        ax.table(cellText = np.round(matrix,2), colLabels = col_names, rowLabels = row_names, loc='center')

        ax.axis('off')
        ax.grid('off')       
        fig.savefig(os.path.join(folder, "last_values"), bbox_inches="tight", dpi=self.resolution)
        plt.close(fig)

    def _save_grid_images(self, this_episode, last_frames=5, folder=None):
        """
        Saves grid with values from the specified last last_frames
        """

        images = []
        obs = copy.deepcopy(this_episode.observations)
        for fr in range(last_frames):
            file_name = "grid_frame{}".format(fr)
            path_name = os.path.join(folder, file_name)
            self._plot_grid(this_episode, obs[-last_frames + fr], path=path_name)
            images.append(imageio.imread("{}.png".format(path_name)))
        
        kwargs = {'duration': 0.8}
        imageio.mimsave(os.path.join(folder, "animated_grid.gif"), images, format='GIF', **kwargs)
    
    
    def _get_all_values(self, observations, str_function):
        ans = []
        for i in range(1,len(observations)):
            ans.append(getattr(observations[i], str_function))
        return np.array(ans)
    
    def _get_all_action_values(self, actions, action_key):
        ans = []
        for i in range(len(actions)):
            act = actions[i].as_dict()[action_key]
            ans.append(act)
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

    test_case = 'tTwo_sandbox'

    agent1 = "{}_DoNothing".format(test_case)
    agent2 = "{}_D3QN".format(test_case)
    agent3 = "{}_MyPTDFAgent".format(test_case)
    #ag_paths = [agent2] # agent1, agent3
    ag_paths = ["tZero_sandbox_D3QN", "tThree_sandbox_D3QN"]
    short_names = ["D3QN (zero)", "D3QN (three)"] # "DoNothing", "ExpertSystem"

    path_save = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases'
    rev = Reviewer(path_save, ag_paths, name='bestDqns', short_names=short_names)

    rev.resume_episodes() # do a graph for all agents
    
    rev.analise_episode("0000")
    rev.analise_episode("0001")
    rev.analise_episode("0002")


