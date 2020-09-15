from MyReviewer import Reviewer

if __name__ == "__main__":

    # test_case = 'tTwo_sandbox'

    # agent1 = "{}_DoNothing".format(test_case)
    # agent2 = "{}_D3QN".format(test_case)
    # agent3 = "{}_MyPTDFAgent".format(test_case)
    #ag_paths = [agent2] # agent1, agent3
    #ag_paths = ["yZero_sandbox_D3QN", "yRZero_sandbox_D3QN", "yTZero_sandbox_D3QN"]

    analysis_folder_name = "just_ExpertSystem" # CHANGE ITTTTTTTT
    ag_paths = ["_AGC_sandbox_ExpertSystem_Di24Reward_(AC_ExpSys_ovf.95)"]
    short_names = ["ExpertSystem"] # "DoNothing", "ExpertSystem"
    notes = [
        "Just reviewing"
        ]
    selected_episodes = ["0002", "0003", "0007", "0011", "0012", "0017", "0018"]


    run_path = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases'
    analysis_path = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases\\_ANLYS_{}'.format(analysis_folder_name)
    rev = Reviewer(run_path, analysis_path, ag_paths, name='a', short_names=short_names, notes=notes, self_analysis=True)

    rev.resume_episodes() # do a graph for all agents    
    for ep in selected_episodes:
        rev.analise_episode(ep)
