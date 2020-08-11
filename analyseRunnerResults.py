import copy
import numpy as np
import matplotlib.pyplot as plt
from grid2op.Episode import EpisodeData
from grid2op.PlotGrid import PlotMatplot

def process_observation(this_episode, type='gen_id', id=0, var='p'):
    if type in ["load_id", "gen_id", "line_id", "substation_id"]:
        dict_params = {type: id}
    else:
        raise Exception("only 'ge', 'li' or 'lo' accepted")
    
    # extract the data
    value = np.zeros(len(this_episode.observations))
    for i, obs in enumerate(this_episode.observations):
        dict_ = obs.state_of(**dict_params) # which effect has this action action on the substation with id 1
        # other objects are: load_id, gen_id, line_id or substation_id
        # see the documentation for more information.
        value[i] = dict_[var]
    
    return value

def plot_ts(ts_values):
    fig, axs = plt.subplots(1,1)
    axs.plot(range(len(ts_values)), ts_values)
    plt.show()
    plt.close(fig)

def plot_grid(this_episode, obs, show=False, path=None):
    plot_helper = PlotMatplot(observation_space=this_episode.observation_space, width=900, height=600)
    plot_helper._line_bus_radius = 7
    fig = plot_helper.plot_obs(obs)
    if show:
        fig.show()
    if not path is None:
        fig.savefig(path)
    plt.close(fig)

def line_connection_summary(this_episode):
    line_disc = 0
    line_reco = 0
    line_changed = 0
    for act in this_episode.actions:
        dict_ = act.as_dict()
        if "set_line_status" in dict_:
            line_reco += dict_["set_line_status"]["nb_connected"]
            line_disc += dict_["set_line_status"]["nb_disconnected"]
        if "change_line_status" in dict_:
            line_changed += dict_["change_line_status"]["nb_changed"]
    print(f'Total lines set to connected : {line_reco}')
    print(f'Total lines set to disconnected : {line_disc}')
    print(f'Total lines changed: {line_changed}')

def hazard_maintenance_summary(this_episode):
    nb_hazards = 0
    nb_maintenance = 0
    for act in this_episode.env_actions:
        dict_ = act.as_dict() # representation of an action as a dictionnary, see the documentation for more information
        if "nb_hazards" in dict_:
            nb_hazards += 1
        if "nb_maintenance" in dict_:
            nb_maintenance += 1
    print(f'Total hazards : {nb_hazards}')
    print(f'Total maintenances : {nb_maintenance}')

if __name__ == "__main__":
    
    path_agent = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\case14sandbox_v0'
    episode_studied = "0001"
    this_episode = EpisodeData.from_disk(path_agent, episode_studied)

    line_connection_summary(this_episode)
    hazard_maintenance_summary(this_episode)
    #plot_ts(process_observation(this_episode, type='load_id', id=0, var='p'))
    plot_ts(this_episode.rewards)
    obs = copy.deepcopy(this_episode.observations[-1])
    obs.topo_vect[3:9] = [2,2,1,1,2,1]
    plot_grid(this_episode, obs, show=True) #path=path_agent