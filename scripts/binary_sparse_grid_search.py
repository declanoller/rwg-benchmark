import path_utils
import Statistics

N_GEN = 20000
N_TRIALS = 10


############################### Basic runs
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'search_method' : 'sparse_bin_grid_search',
        'N_hidden_layers' : 1,
        'N_hidden_units' : 2,
        'use_bias' : False
    },
    {
        'env_name' : ['MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)



#'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
