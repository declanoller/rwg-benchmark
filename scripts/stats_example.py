import path_utils
import Statistics

'''
For getting statistics with various combos of parameters.
'''

############################### Basic runs
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'N_hidden_units' : 4,
        'use_bias' : False,
        'max_episode_steps' : 200
    },
    {
        'env_name' : ['CartPole-v0', 'Pendulum-v0', 'MountainCar-v0'],
        'N_hidden_layers' : [1, 2],
    },
    N_samples=1000,
    N_episodes=10
)
