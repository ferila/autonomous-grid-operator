from MyReviewer import Reviewer

if __name__ == "__main__":

    # test_case = 'tTwo_sandbox'

    # agent1 = "{}_DoNothing".format(test_case)
    # agent2 = "{}_D3QN".format(test_case)
    # agent3 = "{}_MyPTDFAgent".format(test_case)
    #ag_paths = [agent2] # agent1, agent3
    #ag_paths = ["yZero_sandbox_D3QN", "yRZero_sandbox_D3QN", "yTZero_sandbox_D3QN"]

    analysis_folder_name = "good_Di24-DoNg" # CHANGE ITTTTTTTT
    ag_paths = ["_AGC_sandbox_D3QN_Di24Reward_(AC_pqp)", "_AGC_sandbox_DoNothing_Di24Reward_(AC_pqp)"]
    short_names = ["D3QN (Di24)", "DoNothing"] # "DoNothing", "ExpertSystem"
    notes = [
        "The biggest change is to not consider violated values of uncontrollable generators",
        "donothing"
        ]
    selected_episodes = ["0002", "0003", "0007", "0011", "0012", "0017", "0018"]


    run_path = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases'
    analysis_path = 'D:\\ESDA_MSc\\Dissertation\\code_stuff\\cases\\_ANLYS_{}'.format(analysis_folder_name)
    rev = Reviewer(run_path, analysis_path, ag_paths, name='a', short_names=short_names, notes=notes, self_analysis=True)

    rev.resume_episodes() # do a graph for all agents    
    for ep in selected_episodes:
        rev.analise_episode(ep)
