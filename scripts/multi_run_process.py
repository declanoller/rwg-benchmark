import path_utils
from Evolve import *
from Statistics import *

#dir = '/home/declan/Documents/code/RWG_benchmarking/output/Pendulum-v0_evo_13-08-2019_11-40-40.11'
#replot_evo_dict_from_dir(dir)



'''params_dict_list = [
    {
        'N_hidden_layers' : 0,
        'N_hidden_units' : 2
    },
    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 2
    },

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 4
    },

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 8
    }
]'''

params_dict_list = [
    {
        'N_hidden_layers' : 2,
        'N_hidden_units' : 4
    }
]

'''params_dict_list = [
    {
        'N_hidden_layers' : 0
    }
]'''

'''params_dict_list = [
    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 2
    },

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 4
    },
]'''


'''params_dict_list = [

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 2
    },

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 4
    },

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 8
    }
]'''



arch_dict_list = [

    {
        'N_hidden_layers' : 0
    },

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 2
    },

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 4
    },

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 8
    },

    {
        'N_hidden_layers' : 2,
        'N_hidden_units' : 4
    }
]

env_list = [
    'CartPole-v0',
    'Pendulum-v0',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Acrobot-v1'
]
params_dict_list = []

for env in env_list:

    for arch_dict in arch_dict_list:

        params_dict = arch_dict.copy()
        params_dict['env_name'] = env

        params_dict_list.append(params_dict)


print(params_dict_list)
dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/gaussian'
walk_multi_dir(dir, params_dict_list)
exit()

#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_second_run_8.28.2019_uniform/run_2_rand/Stats_vary_env_name_N_hidden_layers_N_hidden_units_27-08-2019_15-37-54_01HL_248HU'
#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_second_run_8.28.2019_uniform/run_2_rand/Stats_vary_env_name_27-08-2019_21-07-16_2HL_4HU'
#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/Stats_vary_env_name_02-09-2019_19-37-15_nobias_0HU_no_acro'

#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/Stats_vary_env_name_N_hidden_units_02-09-2019_21-09-44_nobias_1HL_24HU_no_acro'
#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/Stats_vary_env_name_02-09-2019_22-59-57_nobias_2HL_4HU_no_acro'
#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/uniform/Stats_vary_env_name_N_hidden_units_03-09-2019_01-30-53_nobias_uniform_1HL_248HU'

stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/uniform/Stats_vary_env_name_03-09-2019_08-52-34_nobias_no_acro_uniform_2HL_4HU'

plot_stats_by_env(stats_dir, params_dict_list)

exit()



stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_first_run_8.21.2019_randn/Stats_vary_env_name_N_hidden_layers_N_hidden_units_22-08-2019_11-53-20_01layers_248units_complete/'

plot_envs_vs_NN_arch(stats_dir)
exit()










stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_first_run_8.21.2019_randn/Stats_vary_env_name_22-08-2019_23-20-54_2layers_4units/'
