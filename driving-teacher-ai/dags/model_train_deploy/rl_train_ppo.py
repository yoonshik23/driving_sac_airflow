import os
import sys
working_dir = os.getcwd()

sys.path.append(working_dir)

from model_train_deploy.lib.simulator_02 import Simulator_01
import pandas as pd
from model_train_deploy.lib.ppo import PPOAgent
import os
import torch

db_info = {'host': '163.152.172.163',
          'port': '5432',
          'db_name': 'postgres',
          'user_name': 'tgsociety',
          'password': 'tgsociety'}

# Hyperparameters
action_dim = 25
hidden_dim = 256
lr = 3e-4
batch_size = 64
num_episodes = 1000
working_dir = os.getcwd()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def rl_train(execution_date):

    execution_date = pd.to_datetime(execution_date)
    simulator = Simulator_01(
                                 simulation_start_dt = execution_date, 
                                 simulation_end_dt = execution_date + pd.Timedelta(value = 730, unit = 'day'),
                                 data_end_date = execution_date,
                                 db_info = db_info,
                                 is_first = True,
                                시군구_ids = ['11220', '11230', '11240'], 차량숫자 = 15)
    init_states = simulator.make_init_states(sample_num = 1000)


    agent = PPOAgent(action_dim = action_dim, hidden_dim = hidden_dim, lr = lr,  device = device, 
                     num_cars = len(simulator.차량데이터))
    # agent.load_model(path = './model/model_'+str(len(simulator.차량데이터))+'.torch')
    
    day_경과 = 0
    
    best_score = 0
    # for 단계 in range(2):
    for _ in range(3):
        state = simulator.reset(선택최적 = 0, 선택rule = 'ruleX')
        done = False
        episode_reward = 0
        agent.clear_memory()
        
        while simulator.simulation_done == False:
            actions = agent.select_action(state)
            next_state, reward, done = simulator.step(actions)
            
            if (done[0] == 1):
                day_경과 += 1
            if (done[0] == 1) & (simulator.simulation_done == False):
                next_state.insert(0, [next_state[0][0], [[] for _ in range(25)]])
                
                if day_경과 %10 == 0:
                    agent.log_q_values(init_states)
                    agent.plot_metrics()

                agent.store_transition((state[i], actions[i], reward[i], next_state.copy()[i], done[i]))
                agent.update()
                next_state.pop(0)
            else:

                agent.store_transition((state[i], actions[i], reward[i], next_state.copy()[i], done[i]))
                
            state = next_state

            episode_reward += reward[0]
    
            if simulator.count_total_수업수 > 2000:
                if best_score <= simulator.count_total_수업수/simulator.count_total_차량수:
                    best_score = simulator.count_total_수업수/simulator.count_total_차량수
                    path = './model_train_deploy/model/'+str(simulator.train_id)
                    os.makedirs(path, exist_ok=True)
                    agent.save_model(path = path + '/best_ppo_model.torch')
                    print('saved best model')
        
        print(f" Reward: {episode_reward}")
