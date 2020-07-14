import path_utils
import pprint as pp
from Statistics import *

'''

    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 2
    },
'''

arch_dict_list = [

    {
        'N_hidden_layers' : 0
    },


    {
        'N_hidden_layers' : 1,
        'N_hidden_units' : 4
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

#print(params_dict_list)
#dir = '/home/declan/Documents/code/RWG_benchmarking/output/no_bias/gaussian'
dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/giuse_first_run_8.21.2019_randn'
#dir = '/home/declan/Documents/code/RWG_benchmarking/output/results_data/xeon'
params_results_dict_list = walk_multi_dir(dir, params_dict_list)

d = {}
for p_r_d in params_results_dict_list:

    if p_r_d['env_name'] not in d.keys():
        d[p_r_d['env_name']] = {}

    param_str = '{}HL'.format(p_r_d['N_hidden_layers'])
    if 'N_hidden_units' in p_r_d.keys():
        param_str += '_{}HU'.format(p_r_d['N_hidden_units'])

    d[p_r_d['env_name']]['env'] = p_r_d['env_name']
    d[p_r_d['env_name']][param_str] = '{:.1f} ({:.1f})'.format(p_r_d['best_score'], p_r_d['percentile_99.9'])
    #d[p_r_d['env_name']][param_str] = '{:.1f}'.format(p_r_d['best_score'])
    #d[p_r_d['env_name']][param_str] = '{:.1f}'.format(p_r_d['percentile_99.9'])



col_names_dict = {
    'Environment' : 'env',
    '0 HL' : '0HL',
    '1 HL, 2 HU' : '1HL_2HU',
    '1 HL, 4 HU' : '1HL_4HU',
    '1 HL, 8 HU' : '1HL_8HU',
    '2 HL, 4 HU' : '2HL_4HU'
}
col_widths_dict = {
    'Environment' : '4.3',
    '0 HL' : '1.0',
    '1 HL, 2 HU' : '1.0',
    '1 HL, 4 HU' : '1.0',
    '1 HL, 8 HU' : '1.0',
    '2 HL, 4 HU' : '1.0',
}

col_names_dict = {k:v for k,v in col_names_dict.items() if v in d[list(d.keys())[0]].keys()}


table_string = ''

table_string += r'''
\begin{table}[h!]
\begin{center}
 \begin{tabular}{'''

#table_string += ' '.join(['c']*len(col_names_dict.keys()))
#table_string += ' | '.join(['p{2cm}']*len(col_names_dict.keys()))

table_string += 'm{{{}cm}} || '.format(col_widths_dict['Environment'])
table_string += ' | '.join(['m{{{}cm}}'.format(col_widths_dict[k]) for k in col_names_dict.keys() if k != 'Environment'])

table_string += r'''}
'''

col_list = [k for k in col_names_dict.keys()]
col_list[0] = r'\centering ' + col_list[0]
table_string += (' & '.join(col_list) + r' \\')
table_string += r'''
\hline\hline

'''

table_dict = d
for i, (env, env_dict) in enumerate(table_dict.items()):

    col_val_list = [env_dict[v] for k,v in col_names_dict.items()]
    col_val_list[0] = r'\texttt{' + col_val_list[0] + r'}'
    table_string += (' & '.join(col_val_list) + r' \\')
    if i != len(table_dict)-1:
        table_string += r'''
\hline
'''


caption = 'Best scores found for each architecture used in each environment (with bias nodes)'
table_string += r'''
\end{tabular}
\caption{}
\label{}
\end{center}
\end{table}
'''

print(table_string)
exit()



#
