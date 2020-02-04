import path_utils
import Statistics
import ray, psutil, time

ray.init(num_cpus=psutil.cpu_count(), ignore_reinit_error=True)
print('\nInitializing ray... Waiting for workers before starting...')
time.sleep(2.0)
print('Starting!\n')


'''
For getting statistics with various combos of parameters.
'''

N_GEN = 10000
N_TRIALS = 20



############################### Basic runs
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'N_hidden_units' : 4,
        'use_bias' : False,
        'env_name' : 'Pendulum-v0',
        'max_episode_steps' : 200
    },
    {
    'N_hidden_layers' : [1, 2],
    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)

exit()

############################### Basic runs
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'N_hidden_layers' : 1,
        'use_bias' : False,
        'env_name' : 'Acrobot-v1'
    },
    {
        'N_hidden_units' : [2, 4, 8]
    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)

############################### Basic runs
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'N_hidden_layers' : 2,
        'N_hidden_units' : 4,
        'use_bias' : False,
        'random_dist' : 'uniform'
    },
    {
        'env_name' : ['CartPole-v0', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0', 'Acrobot-v1']

    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)

exit()


############################### Basic runs
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'N_hidden_layers' : 0,
        'use_bias' : False
    },
    {
        'env_name' : ['CartPole-v0', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)
