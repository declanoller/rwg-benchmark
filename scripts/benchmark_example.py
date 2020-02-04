import path_utils
import Benchmark
import os


'''
For testing various benchmarking examples.

'''

#['CartPole-v0', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
Benchmark.benchmark_envs(
                            ['CartPole-v0', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],
                            N_gen=2000,
                            N_dist=25,
                            NN='FFNN_multilayer',
                            N_hidden_layers=0,
                            N_hidden_units=0,
                            act_fn='linear'
                            )


exit()






Benchmark.benchmark_vary_params(
    {
        'env_name' : 'MountainCar-v0',
        'NN' : 'FFNN_multilayer'
    },
    {
        'N_hidden_units' : [2, 4],
        'N_hidden_layers' : [0, 1],
        'act_fn' : ['tanh', 'relu']
    },
    N_gen=5,
    N_dist=4
)

exit()



Benchmark.benchmark_envs(['CartPole-v0'], N_dist=5, N_gen=100)
