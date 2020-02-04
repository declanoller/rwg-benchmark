import path_utils
from Evolve import *
from Statistics import *


#dir = '/home/declan/Documents/code/RWG_benchmarking/output/Pendulum-v0_evo_13-08-2019_11-40-40.11'
dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/xeon/Stats_vary_env_name_04-09-2019_23-13-55/all_runs/env_name=CartPole-v0_04-09-2019_23-13-56'
#dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/xeon/Stats_vary_env_name_04-09-2019_23-13-55_0HL_nobias/all_runs/env_name=CartPole-v0_04-09-2019_23-13-56'
replot_evo_dict_from_dir(dir, N_bins=40)
exit()


#multi_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_first_run_8.21.2019_randn'

#multi_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/run_1_randn'
multi_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/xeon'
plot_envs_vs_NN_arch(multi_dir, ylabel_prefix='No bias,\n')
#plot_envs_vs_NN_arch(multi_dir, ylabel_prefix='With bias,\n')
exit()





'''
arch_list = [
    {
        'N_hidden_layers' : 0,
        'arch_title' : '0 hidden layers'
    },
    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 2,
        'arch_title' : '1 hidden layers, \n2 hidden units'
    },
    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 4,
        'arch_title' : '1 hidden layers, \n4 hidden units'
    },
    {
        'N_hidden_layers' : 2,
        'N_hidden_units' : 4,
        'arch_title' : '2 hidden layers, \n4 hidden units'
    },
]
'''



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


#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_second_run_8.28.2019_uniform/run_2_rand/Stats_vary_env_name_N_hidden_layers_N_hidden_units_27-08-2019_15-37-54_01HL_248HU'
#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_second_run_8.28.2019_uniform/run_2_rand/Stats_vary_env_name_27-08-2019_21-07-16_2HL_4HU'
#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/Stats_vary_env_name_02-09-2019_19-37-15_nobias_0HU_no_acro'

#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/Stats_vary_env_name_N_hidden_units_02-09-2019_21-09-44_nobias_1HL_24HU_no_acro'
#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/Stats_vary_env_name_02-09-2019_22-59-57_nobias_2HL_4HU_no_acro'
#stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/uniform/Stats_vary_env_name_N_hidden_units_03-09-2019_01-30-53_nobias_uniform_1HL_248HU'




stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/uniform/Stats_vary_env_name_03-09-2019_08-52-34_nobias_no_acro_uniform_2HL_4HU'

plot_stats_by_env(stats_dir, params_dict_list)

exit()













stats_dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_first_run_8.21.2019_randn/Stats_vary_env_name_22-08-2019_23-20-54_2layers_4units/'
