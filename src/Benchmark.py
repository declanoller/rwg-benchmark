import path_utils
from Evolve import Evolve
import traceback as tb
import os, json
import numpy as np
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import pprint as pp
import pandas as pd
from tabulate import tabulate
import seaborn as sns

'''
Will eventually be made into a class that will handle all the benchmarking
details. For now, just holds benchmark_envs.

NOTE: This is probably broken at the moment because I changed a few things about
Evolve.py (output_dir -> run_dir, base_dir, etc).


'''

################################ Benchmark functions

@path_utils.timer
def benchmark_envs(env_list, **kwargs):

    '''
    Iterates over a list of env names you give it,
    benchmarking it and recording info.

    For each env, it
    '''

    N_dist = kwargs.get('N_dist', 10) # How many evolutions to run, to form a distribution
    N_gen = kwargs.get('N_gen', 1000)

    # Create dir for the results of this benchmark.
    benchmark_dir = os.path.join(path_utils.get_output_dir(), 'Benchmark_{}'.format(path_utils.get_date_str()))
    os.mkdir(benchmark_dir)

    # Dict to hold results on timing, etc.
    benchmark_dict = {}

    for env_name in env_list:

        print(f'\nBenchmarking env {env_name} now...\n')
        # Create a dir for this env.
        env_dir = os.path.join(benchmark_dir, env_name)
        os.mkdir(env_dir)

        env_runs_dir = os.path.join(env_dir, 'runs')
        os.mkdir(env_runs_dir)

        param_dict = deepcopy(kwargs)
        param_dict['env_name'] = env_name

        benchmark_dict[env_name] = benchmark_param_dict(param_dict, N_dist, N_gen, env_runs_dir)
        benchmark_dict[env_name]['env_dir'] = env_dir


    # Save distributions to file
    with open(os.path.join(benchmark_dir, 'benchmark_stats.json'), 'w+') as f:
        json.dump(benchmark_dict, f, indent=4)

    # Plot each env dist.
    for k,v in benchmark_dict.items():

        # Makes sure the run finished
        if 'solve_gens' in v.keys():
            fname = os.path.join(v['env_dir'], f'{k}_solve_gens_dist.png')
            plot_benchmark_dist(k, v['solve_gens'], 'Solve generation', fname)

            fname = os.path.join(v['env_dir'], f'{k}_best_scores_dist.png')
            plot_benchmark_dist(k, v['best_scores'], 'Best score', fname)

            fname = os.path.join(v['env_dir'], f'{k}_all_scores_dist.png')
            plot_benchmark_dist(k, v['all_scores'], 'All scores', fname, N_bins=20, plot_log=True)




def benchmark_param_dict(param_dict, N_dist, N_gen, output_dir):

    '''
    Pass it a single dict with the params to benchmark, including the env_name.

    Also pass an output_dir, or it will use the default output folder.

    Note: for each dict, it does an evolution MULTIPLE times.
    '''

    # deepcopy, just to be safer
    params = deepcopy(param_dict)

    assert 'env_name' in params.keys(), 'Must supply an env_name!'

    env_name = params['env_name']
    params.pop('env_name')
    params['output_dir'] = output_dir

    solve_gen_dist = [] # To hold the solve times distribution for this env.
    best_score_dist = [] # To hold the solve times distribution for this env.
    all_scores_dist = []
    try:

        for dist_run in range(N_dist):
            print(f'evolution {dist_run + 1}/{N_dist}')

            e = Evolve(env_name, **params)
            evo_dict = e.evolve(N_gen)
            e.plot_scores(evo_dict)
            # If it didn't solve it, give it the max number (which
            # may actually still be an underestimate).
            if evo_dict['solved']:
                solve_gen_dist.append(evo_dict['solve_gen'])
            else:
                solve_gen_dist.append(N_gen)

            best_score_dist.append(max(evo_dict['best_scores']))

            all_scores_dist += evo_dict['all_scores']

        return {
            'solve_gens' : solve_gen_dist,
            'best_scores' : best_score_dist,
            'all_scores' : all_scores_dist
        }

    except:
        print(f'\n\nError in evolve with params: {params}. Traceback:\n')
        print(tb.format_exc())
        print('\n\nAttempting to continue...\n\n')

        return {}


def benchmark_classic_control_envs(**kwargs):

    '''
    Loads gym_envs_info.json. This contains info about the envs we want to benchmark.

    It then calls benchmark_envs() for the classic control envs.
    '''

    with open(os.path.join(path_utils.get_src_dir(), 'gym_envs_info.json'), 'r') as f:
        envs_dict = json.load(f)

    env_list = [k for k,v in envs_dict.items() if v['env_type']=='classic_control']

    print(f'Benchmarking: {env_list}')

    benchmark_envs(env_list, **kwargs)



def benchmark_param_dicts(params_dict_list, **kwargs):

    '''
    Pass this a list of dicts, where each has the different parameters you want
    to benchmark.

    It then iterates through this list, doing a benchmark for each dict.
    '''

    # Create dir for the results of this benchmark if one isn't provided.
    benchmark_dir = kwargs.get('benchmark_dir', None)
    if benchmark_dir is None:
        benchmark_dir = os.path.join(path_utils.get_output_dir(), 'Benchmark_{}'.format(path_utils.get_date_str()))
        os.mkdir(benchmark_dir)

    for d in params_dict_list:

        # If a run_fname_label is provided, use that to create a more informative dir name.
        # Otherwise, just use the date.
        if 'run_fname_label' in d.keys():
            run_fname_label = d['run_fname_label']
        else:
            run_fname_label = 'vary_params'

        # Base dir for this specific benchmark
        params_dir = os.path.join(benchmark_dir, '{}_{}'.format(run_fname_label, path_utils.get_date_str()))
        os.mkdir(params_dir)

        # To hold the actual runs (FF)
        runs_dir = os.path.join(params_dir, 'runs')
        os.mkdir(runs_dir)

        print('\n\nNow benchmarking params:')
        pp.pprint(d, width=1)
        print('\n\n')
        benchmark_dict = benchmark_param_dict(d, kwargs.get('N_dist', 10), kwargs.get('N_gen', 100), runs_dir)

        # Add to dict
        d['benchmark_dict'] = deepcopy(benchmark_dict)

        # Make plots for this benchmark
        if 'run_plot_label' in d.keys():
            run_plot_label = d['run_plot_label']
        else:
            run_plot_label = run_fname_label

        # Plots for benchmark
        fname = os.path.join(params_dir, f'{run_fname_label}_solve_gens_dist.png')
        plot_benchmark_dist(run_plot_label, benchmark_dict['solve_gens'], 'Solve generation', fname)
        fname = os.path.join(params_dir, f'{run_fname_label}_best_scores_dist.png')
        plot_benchmark_dist(run_plot_label, benchmark_dict['best_scores'], 'Best score', fname)

    # Return passed list, which should have dicts
    # modified with the results
    return params_dict_list


@path_utils.timer
def benchmark_vary_params(constant_params_dict, vary_params_dict, **kwargs):

    '''
    This is a convenience function to easily vary parameters for benchmarking.
    You pass it constant_params_dict, which is a dict with the values that
    you want to remain constant between runs. Then, pass it vary_params_dict,
    which should have each parameter that you want to vary as a list of the values
    it should take.

    Example:

    constant_params_dict = {
        'env_name' : 'CartPole-v0',
        'N_gen' : 1000,
        'N_dist' : 100,
        'NN' : 'FFNN_multilayer'
    }

    vary_params_dict = {
        'N_hidden_units' : [2, 4, 8],
        'act_fn' : ['tanh', 'relu']
    }

    This will do 3*2 = 6 runs, for each of the combinations of varying parameters.
    '''

    # Create informative dir name
    vary_params = list(vary_params_dict.keys())
    benchmark_dir = os.path.join(
                        path_utils.get_output_dir(),
                        'Benchmark_vary_{}_{}'.format('_'.join(vary_params), path_utils.get_date_str()))
    print(f'\nSaving benchmark run to {benchmark_dir}')
    os.mkdir(benchmark_dir)

    combined_params = {**constant_params_dict, **vary_params_dict}
    # Save params to file
    with open(os.path.join(benchmark_dir, 'run_params.json'), 'w+') as f:
        json.dump(combined_params, f, indent=4)


    # Flatten list, pass to other function
    flat_param_list = vary_params_cross_products(constant_params_dict, vary_params_dict)
    flat_param_list = benchmark_param_dicts(flat_param_list, benchmark_dir=benchmark_dir, **kwargs)

    # Parse results
    for d in flat_param_list:
        benchmark_dict = d['benchmark_dict']

        best_scores = benchmark_dict['best_scores']
        d['mu_best'] = np.mean(best_scores)
        d['sigma_best'] = np.std(best_scores)

        solve_gens = benchmark_dict['solve_gens']
        d['mu_solve_gens'] = np.mean(solve_gens)
        d['sigma_solve_gens'] = np.std(solve_gens)

        #pp.pprint(d, width=1)
        # Get rid of this now
        d.pop('benchmark_dict')

    # Save results to csv for later parsing/plotting
    df = pd.DataFrame(flat_param_list)
    print(tabulate(df, headers=df.columns.values, tablefmt='psql'))
    df_fname = os.path.join(benchmark_dir, 'vary_benchmark_results.csv')
    df.to_csv(df_fname, index=False)

    # Only need to do if more than 2 params were varied.
    if len(vary_params) >= 2:

        # Create heatmap plots dir
        heatmap_dir = os.path.join(benchmark_dir, 'heatmap_plots')
        print(f'\nSaving heatmap plots to {heatmap_dir}')
        os.mkdir(heatmap_dir)

        # Iterate over all unique pairs of vary params, plot heatmaps of them
        for pair in itertools.combinations(vary_params, 2):

            print(f'Making heatmaps for {pair}')

            other_params_flat = [(k, v) for k,v in vary_params_dict.items() if k not in pair]
            other_params = [x[0] for x in other_params_flat]
            other_vals = [x[1] for x in other_params_flat]
            print(f'other params: {other_params}')

            # Create dir for specific pivot
            pivot_name = 'vary_{}_{}'.format(*pair)
            pivot_dir = os.path.join(heatmap_dir, pivot_name)
            os.mkdir(pivot_dir)

            # Select for each of the combos of the other params.
            for other_params_set in itertools.product(*other_vals):
                other_sel_dict = dict(zip(other_params, other_params_set))
                fname_label = path_utils.param_dict_to_fname_str(other_sel_dict)
                df_sel = df.loc[(df[list(other_sel_dict)] == pd.Series(other_sel_dict)).all(axis=1)]

                heatmap_plot(df_sel, *pair, 'mu_best', pivot_dir, label=fname_label)
                heatmap_plot(df_sel, *pair, 'mu_solve_gens', pivot_dir, label=fname_label)






def heatmap_plot(df, xvar, yvar, zvar, output_dir, **kwargs):

    #df = pd.read_csv(csv_fname)
    df = df.pivot(yvar, xvar, zvar)


    plt.close('all')
    plt.figure()
    ax = plt.gca()

    label = kwargs.get('label', '')

    sns.heatmap(df, annot=True, fmt=".1f", cmap='viridis', ax=ax)
    ax.set_title(f'{zvar} for constant {label}')
    plt.savefig(os.path.join(output_dir, f'vary_{xvar}_{yvar}__{zvar}_heatmap__const_{label}.png'))
    if kwargs.get('show_plot', False):
        plt.show()





################################# Helper functions

def plot_benchmark_dist(run_fname_label, dist, dist_label, fname, **kwargs):

    '''
    For plotting the distribution of various benchmarking stats for the run_fname_label.
    Plots a vertical dashed line at the mean.
    '''

    plt.close('all')
    mu = np.mean(dist)
    sd = np.std(dist)

    if kwargs.get('N_bins', None) is None:
        plt.hist(dist, color='dodgerblue', edgecolor='gray')
    else:
        plt.hist(dist, color='dodgerblue', edgecolor='gray', bins=kwargs.get('N_bins', None))

    plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
    plt.xlabel(dist_label)
    plt.ylabel('Counts')
    plt.title(f'{dist_label} distribution for {run_fname_label}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$')
    plt.savefig(fname)

    if kwargs.get('plot_log', False):
        if kwargs.get('N_bins', None) is None:
            plt.hist(dist, color='dodgerblue', edgecolor='gray', log=True)
        else:
            plt.hist(dist, color='dodgerblue', edgecolor='gray', bins=kwargs.get('N_bins', None), log=True)

        plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
        plt.xlabel(dist_label)
        plt.ylabel('log(Counts)')
        plt.title(f'{dist_label} distribution for {run_fname_label}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$')
        plt.savefig(fname.replace('dist', 'log_dist'))


def vary_params_cross_products(constant_params_dict, vary_params_dict):

    '''
    Gets and returns the "cross product" of several lists in vary_params_dict,
    as a large flat dict.

    '''

    param_dict_list = []

    # Do this because we want to make sure to preserve order. Might not be necessary?
    vary_params_flat = [(k, v) for k,v in vary_params_dict.items()]
    vary_args = [x[0] for x in vary_params_flat]
    vary_vals = [x[1] for x in vary_params_flat]

    # Iterate over all combinations of vary params.
    for cur_vals in itertools.product(*vary_vals):

        current_vary_dict = dict(zip(vary_args, cur_vals))

        full_params = {**deepcopy(constant_params_dict), **current_vary_dict}

        # Get a run label, will be useful for varying lots.
        full_params['run_fname_label'] = path_utils.param_dict_to_fname_str(current_vary_dict)
        full_params['run_plot_label'] = path_utils.linebreak_every_n_spaces(
                                            path_utils.param_dict_to_label_str(
                                                            current_vary_dict))

        param_dict_list.append(full_params)

    return param_dict_list






#
