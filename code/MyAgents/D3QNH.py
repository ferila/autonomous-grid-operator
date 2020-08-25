import numpy as np
from DQNb import MyDoubleDuelingDQN

class MyDoubleDuelingDQN_H(MyDoubleDuelingDQN):

    def __init__(self, observation_space, action_space, name=__name__, is_training=False, num_frames=4, batch_size=32, lr=1e-5, topology_heur=None):
        super().__init__(observation_space, 
                        action_space, 
                        name=name, 
                        is_training=is_training, 
                        num_frames=num_frames, 
                        batch_size=batch_size, 
                        lr=lr)

        self.tph = topology_heur
    
    def convert_obs(self, observation):
        res = []
        
        # include generation observations
        gen = observation.prod_p / observation.gen_pmax
        res.append(gen)

        # include powerflow observations
        powerflow_limit = 2 # 200% of powerflow disconnects automatically the powerline
        max_val = powerflow_limit
        min_val = -powerflow_limit
        pflow = (np.sign(observation.p_or) * observation.rho - min_val) / (max_val - min_val)
        res.append(pflow)

        # include time (year not included, should work only on long term)
        res.append(np.array([observation.minute_of_hour / 60]))
        res.append(np.array([observation.hour_of_day / 24]))
        res.append(np.array([observation.day_of_week / 7]))
        #res.append(observation.month / 12)

        # include prior flows
        ptdf = self.tph_ptdf.check_calculated_matrix(observation.line_status)        


        return np.concatenate(res)