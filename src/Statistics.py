import path_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import traceback as tb
import os, json, shutil, time
import itertools
from copy import deepcopy
import pprint as pp
from tabulate import tabulate

from Sample import Sample, replot_sample_dict_from_dir


'''

For running RWG for various combinations of architectures and collecting statistics.

run_param_dict() is the most basic function, just doing a sampling for a passed
param_dict. Other functions basically involve calling it given various inputs.

'''

################################ Statistics functions


@path_utils.timer
def run_param_dict(param_dict, N_samples, N_episodes, base_dir):

    '''
    Pass a single params dict to run an sample() of, including the env_name.

    Also pass an output_dir, or it will use the default output folder.

    This only runs each setting ONCE.
    '''

    # deepcopy, just to be safer
    params = deepcopy(param_dict)

    assert 'env_name' in params.keys(), 'Must supply an env_name!'

    env_name = params['env_name']
    params.pop('env_name')
    params['base_dir'] = base_dir

    try:
        # Run a single parameters setting
        e = Sample(env_name, **params)
        sample_dict = e.sample(N_samples, N_episodes=N_episodes, print_samp_num=True)
        e.save_all_sample_stats(sample_dict, save_plots=True)

        return sample_dict

    except:
        print(f'\n\nError in sample() with params: {params}. Traceback:\n')
        print(tb.format_exc())
        print('\n\nAttempting to continue...\n\n')

        return {}


def run_param_dict_wrapper(param_dict, N_samples, N_episodes, base_dir):

    # If a run_fname_label is provided, use that to create a more informative dir name.
    # Otherwise, just use the date.
    if 'run_fname_label' in param_dict.keys():
        run_fname_label = param_dict['run_fname_label']
    else:
        run_fname_label = 'vary_params'

    # Make plots for this params set
    if 'run_plot_label' in param_dict.keys():
        run_plot_label = param_dict['run_plot_label']
    else:
        run_plot_label = run_fname_label

    # Run dir for this set of params
    params_dir = os.path.join(base_dir, '{}_{}'.format(run_fname_label, path_utils.get_date_str()))
    os.mkdir(params_dir)
    # Doing this so it just saves directly to this dir, which has a more
    # informative name than Sample.__init__() would create.
    param_dict['run_dir'] = params_dir

    print('\n\nNow running with params:')
    pp.pprint(param_dict, width=1)
    print('\n\n')

    stats_dict = run_param_dict(param_dict, N_samples, N_episodes, base_dir)
    return stats_dict



@path_utils.timer
def run_multi_envs(env_list, **kwargs):

    '''
    Iterates over a list of env names you give it,
    running them and recording info.
    '''

    N_samples = kwargs.get('N_samples', 1000)
    N_episodes = kwargs.get('N_episodes', 1000)

    # Create dir for the results of this stats set.
    stats_dir = os.path.join(path_utils.get_output_dir(), 'Stats_{}'.format(path_utils.get_date_str()))
    os.mkdir(stats_dir)

    # Dict to hold results on timing, etc.
    stats_dict = {}

    for env_name in env_list:

        print(f'\nGetting stats for env {env_name} now...\n')

        param_dict = deepcopy(kwargs)
        param_dict['env_name'] = env_name

        stats_dict[env_name] = run_param_dict(param_dict, N_samples, N_episodes, stats_dir)


    # Save distributions to file
    with open(os.path.join(stats_dir, 'multi_env_stats.json'), 'w+') as f:
        json.dump(stats_dict, f, indent=4)



def run_classic_control_envs(**kwargs):

    '''
    Loads gym_envs_info.json. This contains info about the envs we want to analyze.

    It then calls run_multi_envs() for the classic control envs.
    '''

    with open(os.path.join(path_utils.get_src_dir(), 'gym_envs_info.json'), 'r') as f:
        envs_dict = json.load(f)

    env_list = [k for k,v in envs_dict.items() if v['env_type']=='classic_control']

    print(f'Getting stats for: {env_list}')

    run_multi_envs(env_list, **kwargs)


def run_param_dict_list(params_dict_list, **kwargs):

    '''
    Pass this a list of dicts, where each has the different parameters you want
    to gather stats for.

    It then iterates through this list, doing a run for each dict.

    Note that it modifies the passed params_dict_list to add the results to it.

    '''

    # Create dir for the results of this stats run if one isn't provided.
    stats_dir = kwargs.get('stats_dir', None)
    if stats_dir is None:
        stats_dir = os.path.join(path_utils.get_output_dir(), 'Stats_{}'.format(path_utils.get_date_str()))
        os.mkdir(stats_dir)

    # Produce results in parallel
    for d in params_dict_list:
        # For non-ray use
        d['result'] = run_param_dict_wrapper(   d,
                                                kwargs.get('N_samples', 100),
                                                kwargs.get('N_episodes', 10),
                                                stats_dir)
        # For use with ray
        '''d['result_ID'] = run_param_dict_wrapper.remote( d,
                                                        kwargs.get('N_samples', 100),
                                                        kwargs.get('N_episodes', 10),
                                                        stats_dir)'''

    # Retrieve results from ID
    for d in params_dict_list:
        d['stats_dict'] = d['result'] # for non-ray use
        d.pop('result')

    # Return passed list, which should have dicts
    # modified with the results
    return params_dict_list



@path_utils.timer
def run_vary_params(constant_params_dict, vary_params_dict, **kwargs):

    '''
    This is a convenience function to easily vary parameters for analysis.
    You pass it constant_params_dict, which is a dict with the values that
    you want to remain constant between runs. Then, pass it vary_params_dict,
    which should have each parameter that you want to vary as a list of the values
    it should take.

    Example:

    constant_params_dict = {
        'env_name' : 'CartPole-v0',
        'N_samples' : 1000,
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
    stats_dir = os.path.join(
                        path_utils.get_output_dir(),
                        'Stats_vary_{}_{}'.format('_'.join(vary_params), path_utils.get_date_str()))
    print(f'\nSaving statistics run to {stats_dir}')
    os.mkdir(stats_dir)

    # Create runs dir
    all_runs_dir = os.path.join(stats_dir, 'all_runs')
    print(f'\nSaving all runs to {all_runs_dir}')
    os.mkdir(all_runs_dir)

    # Create dict of const and vary params, as separate items
    all_params = {
        'const_params' : constant_params_dict,
        'vary_params' : vary_params_dict
    }
    other_run_params = ['N_samples', 'N_episodes']
    for p in other_run_params:
        if p in kwargs.keys():
            all_params[p] = kwargs.get(p, None)

    # Save params to file
    with open(os.path.join(stats_dir, 'all_params.json'), 'w+') as f:
        json.dump(all_params, f, indent=4)


    # Flatten list, pass to other function
    flat_param_list = vary_params_cross_products(constant_params_dict, vary_params_dict)
    flat_param_list = run_param_dict_list(flat_param_list, stats_dir=all_runs_dir, **kwargs)

    # Parse results
    for d in flat_param_list:
        # For now I'll still keep vary_params_stats.csv, but I think it's not
        # actually necessary.
        # Get rid of this now
        d.pop('stats_dict')

    # Save results to csv for later parsing/plotting
    df = pd.DataFrame(flat_param_list)
    print(tabulate(df, headers=df.columns.values, tablefmt='psql'))
    df_fname = os.path.join(stats_dir, 'vary_params_stats.csv')
    df.to_csv(df_fname, index=False)




################################# Plotting functions


def plot_all_agg_stats(stats_dir):

    '''
    For plotting all the heatmaps/etc for a stats_dir.

    '''

    agg_stats_dir = os.path.join(stats_dir, 'agg_stats')
    if os.path.exists(agg_stats_dir):
        shutil.rmtree(agg_stats_dir)
    print(f'\nSaving all aggregate stats to {agg_stats_dir}')
    os.mkdir(agg_stats_dir)


    all_params_fname = os.path.join(stats_dir, 'all_params.json')
    with open(all_params_fname, 'r') as f:
        all_params_dict = json.load(f)

    # Import all scores
    all_scores_fname = os.path.join(stats_dir, 'all_scores.csv')
    df = pd.read_csv(all_scores_fname)

    vary_params_dict = all_params_dict['vary_params']
    const_params_dict = all_params_dict['const_params']
    vary_params = list(vary_params_dict.keys())
    N_vary_params = len(vary_params)


    # Only need to do if more than 2 params were varied.
    if N_vary_params >= 2:

        # Get envs info to find out percent of runs that "solved" the env.
        with open(os.path.join(path_utils.get_src_dir(), 'gym_envs_info.json'), 'r') as f:
            envs_dict = json.load(f)

        # Create heatmap plots dir
        heatmap_dir = os.path.join(agg_stats_dir, 'heatmap_plots')
        if os.path.exists(heatmap_dir):
            shutil.rmtree(heatmap_dir)
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
                # This part just selects for the rows that have the correct
                # params/etc.
                other_sel_dict = dict(zip(other_params, other_params_set))
                fname_label = path_utils.param_dict_to_fname_str(other_sel_dict)
                df_sel = df.loc[(df[list(other_sel_dict)] == pd.Series(other_sel_dict)).all(axis=1)]

                df_no_scores = df_sel.drop('all_scores', axis=1)
                #print(df_no_scores.columns.values)

                df_params_only = df_no_scores.drop_duplicates()

                all_row_dfs = []

                # Iterate through table, for each run label, find its corresponding dir,
                # walk through it, get all its scores, create a dataframe from them,
                # then concatenate all these df's into a big one, that we can plot.
                for index, row in df_params_only.iterrows():
                    # Only get the params varied, turn them into a dict

                    row_dict = row[df_no_scores.columns.values].to_dict()

                    df_row = df_sel.loc[(df[list(row_dict)] == pd.Series(row_dict)).all(axis=1)]
                    row_scores = df_row['all_scores'].values


                    row_dict['index'] = index
                    row_dict['mean_score'] = np.mean(row_scores)
                    row_dict['best_score'] = np.max(row_scores)

                    solved_reward = envs_dict[row_dict['env_name']]['solved_avg_reward']
                    N_solved_scores = np.sum(np.where(row_scores >= solved_reward))
                    row_dict['percent_solved_scores'] = N_solved_scores/len(row_scores)

                    # pandas has the nice perk that if you create a df from a dict where
                    # some of the entries are constants and one entry is a list, it duplicates
                    # the constant values.
                    row_df = pd.DataFrame(row_dict, index=[index])
                    all_row_dfs.append(row_df)


                all_scores_df = pd.concat(all_row_dfs)

                #print(tabulate(all_scores_df, headers=all_scores_df.columns.values, tablefmt='psql'))

                heatmap_plot(all_scores_df, *pair, 'mean_score', pivot_dir, label=fname_label)
                heatmap_plot(all_scores_df, *pair, 'best_score', pivot_dir, label=fname_label)
                heatmap_plot(all_scores_df, *pair, 'percent_solved_scores', pivot_dir, label=fname_label)
                #heatmap_plot(df_sel, *pair, perc_cutoff_str, pivot_dir, label=fname_label)




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



def make_total_score_df(stats_dir):

    # Load csv that holds the names of all the dirs
    stats_overview_fname = os.path.join(stats_dir, 'vary_params_stats.csv')
    overview_df = pd.read_csv(stats_overview_fname)

    # unique fname labels in table
    run_fname_labels = overview_df.run_fname_label.unique()

    # Get the params that are varied
    all_params_fname = os.path.join(stats_dir, 'all_params.json')
    with open(all_params_fname, 'r') as f:
        all_params_dict = json.load(f)

    const_params = list(all_params_dict['const_params'].keys())
    vary_params = list(all_params_dict['vary_params'].keys())
    print(f'Params varied: {vary_params}')

    all_row_dfs = []
    runs_dir = os.path.join(stats_dir, 'all_runs')
    # Iterate through table, for each run label, find its corresponding dir,
    # walk through it, get all its scores, create a dataframe from them,
    # then concatenate all these df's into a big one, that we can plot.
    for index, row in overview_df.iterrows():
        # Only get the params varied, turn them into a dict
        #row_dict = row[vary_params].to_dict()
        row_dict = row[vary_params + const_params].to_dict()
        #row_dict = row.to_dict()
        run_label = row['run_fname_label']
        # Get the one dir that has the run_fname_label in its name
        match_dirs = [x for x in os.listdir(runs_dir) if run_label in x]
        assert len(match_dirs)==1, 'Must only have one dir matching label!'
        vary_dir = match_dirs[0]
        # Clumsy, but: walk through this dir until you find the sample_stats.json,
        # then add its scores to the row_dict
        for root, dirs, files in os.walk(os.path.join(runs_dir, vary_dir)):
            if 'sample_stats.json' in files:
                with open(os.path.join(root, 'sample_stats.json'), 'r') as f:
                    sample_dict = json.load(f)

                row_dict['all_scores'] = sample_dict['all_scores']

        # pandas has the nice perk that if you create a df from a dict where
        # some of the entries are constants and one entry is a list, it duplicates
        # the constant values.
        row_df = pd.DataFrame(row_dict)
        all_row_dfs.append(row_df)


    all_scores_df = pd.concat(all_row_dfs)
    all_scores_fname = os.path.join(stats_dir, 'all_scores.csv')
    all_scores_df.to_csv(all_scores_fname, index=False)



def violin_plot(df, xvar, yvar, output_dir, **kwargs):

    #df = pd.read_csv(csv_fname)
    #df = df.pivot(yvar, xvar, zvar)


    plt.close('all')
    plt.figure()
    ax = plt.gca()

    label = kwargs.get('label', '')
    sns.violinplot(x=xvar, y="all_scores", hue=yvar, data=df)

    #sns.heatmap(df, annot=True, fmt=".1f", cmap='viridis', ax=ax)
    ax.set_title(f'all_scores for constant {label}')
    plt.savefig(os.path.join(output_dir, f'vary_{xvar}_{yvar}__all_scores_violinplot__const_{label}.png'))
    if kwargs.get('show_plot', False):
        plt.show()


def all_violin_plots(stats_dir):

    all_scores_fname = os.path.join(stats_dir, 'all_scores.csv')

    if not os.path.exists(all_scores_fname):
        make_total_score_df(stats_dir)

    # Load csv that holds the names of all the dirs
    df = pd.read_csv(all_scores_fname)

    # Get the params that are varied
    vary_params_fname = os.path.join(stats_dir, 'all_params.json')
    with open(vary_params_fname, 'r') as f:
        all_params_dict = json.load(f)

    vary_params = [k for k,v in all_params_dict.items() if isinstance(v, list)]
    vary_params_dict = {k:v for k,v in all_params_dict.items() if isinstance(v, list)}

    print(f'Params varied: {vary_params}')


    # Only need to do if more than 2 params were varied.
    if len(vary_params) >= 2:

        # Create violin plots dir
        violin_plots_dir = os.path.join(stats_dir, 'violin_plots')
        print(f'\nSaving violin plots to {violin_plots_dir}')
        if os.path.exists(violin_plots_dir):
            shutil.rmtree(violin_plots_dir)
        os.mkdir(violin_plots_dir)

        # Iterate over all unique pairs of vary params, plot violin plots of them
        for pair in itertools.combinations(vary_params, 2):

            print(f'Making violin plots for {pair}')

            other_params_flat = [(k, v) for k,v in vary_params_dict.items() if k not in pair]
            other_params = [x[0] for x in other_params_flat]
            other_vals = [x[1] for x in other_params_flat]
            print(f'other params: {other_params}')

            # Create dir for specific pivot
            pivot_name = 'vary_{}_{}'.format(*pair)
            pivot_dir = os.path.join(violin_plots_dir, pivot_name)
            os.mkdir(pivot_dir)

            # Select for each of the combos of the other params.
            for other_params_set in itertools.product(*other_vals):
                other_sel_dict = dict(zip(other_params, other_params_set))
                fname_label = path_utils.param_dict_to_fname_str(other_sel_dict)
                df_sel = df.loc[(df[list(other_sel_dict)] == pd.Series(other_sel_dict)).all(axis=1)]
                print(other_sel_dict)
                violin_plot(df_sel, *pair, pivot_dir, label=fname_label)



def plot_stats_by_env(stats_dir, params_dict_list):

    all_params_fname = os.path.join(stats_dir, 'all_params.json')
    assert os.path.exists(all_params_fname), f'all_params.json file {all_params_fname} DNE!'
    with open(all_params_fname, 'r') as f:
        all_params_dict = json.load(f)

    # Load csv that holds the names of all the dirs
    stats_overview_fname = os.path.join(stats_dir, 'vary_params_stats.csv')
    overview_df = pd.read_csv(stats_overview_fname)
    drop_list = ['result_ID', 'run_plot_label', 'mu_all_scores', 'sigma_all_scores', 'all_scores_99_perc']
    for d in drop_list:
        if d in overview_df.columns.values:
            overview_df = overview_df.drop(d, axis=1)

    for params_dict in params_dict_list:
        plot_2x2_grids(stats_dir, overview_df, params_dict)








def plot_2x2_grids(stats_dir, overview_df, params_dict, **kwargs):

    params_fname_label = '_'.join([f'{k}={v}' for k,v in params_dict.items()])

    figures_base_dir = os.path.join(path_utils.get_output_dir(), 'figures')
    figures_dir = os.path.join(figures_base_dir, '2x2_and_single')
    params_dir = os.path.join(figures_dir, params_fname_label)

    if os.path.exists(params_dir):
        shutil.rmtree(params_dir)

    os.mkdir(params_dir)

    for k, v in params_dict.items():
        subset_df = overview_df[(overview_df[k] == v)]


    #print(tabulate(subset_df, headers=subset_df.columns.values, tablefmt='psql'))

    envs_list = [
        'MountainCarContinuous-v0',
        'CartPole-v0',
        'Acrobot-v1',
        'MountainCar-v0',
        'Pendulum-v0'
    ]

    all_runs_dir = os.path.join(stats_dir, 'all_runs')

    env_score_dict = {}

    for env in envs_list:

        env_score_dict[env] = None
        if env in subset_df['env_name'].values:
            run_fname_label = subset_df[subset_df['env_name']==env]['run_fname_label'].values[0]

            match_dirs = [x for x in os.listdir(all_runs_dir) if run_fname_label in x]
            assert len(match_dirs)==1, 'Must only have one dir matching label!'
            vary_dir = match_dirs[0]
            # Clumsy, but: walk through this dir until you find the sample_stats.json,
            # then add its scores to the row_dict
            for root, dirs, files in os.walk(os.path.join(all_runs_dir, vary_dir)):
                if 'sample_stats.json' in files:
                    with open(os.path.join(root, 'sample_stats.json'), 'r') as f:
                        sample_dict = json.load(f)

                    env_score_dict[env] = sample_dict['all_trials']


    plot_pt_alpha = 0.2
    plot_label_params = {
        'fontsize' : 10
    }
    plot_tick_params = {
        'axis' : 'both',
        'labelsize' : 8
    }
    plot_title_params = {
        'fontsize' : 12
    }
    solo_plot_label_params = {
        'fontsize' : 10
    }
    solo_plot_tick_params = {
        'axis' : 'both',
        'labelsize' : 8
    }
    solo_plot_title_params = {
        'fontsize' : 12
    }


    solo_env = 'CartPole-v0'
    grid_envs = [
        'MountainCarContinuous-v0',
        'Acrobot-v1',
        'MountainCar-v0',
        'Pendulum-v0'
    ]
    env_score_dict_grid = {k:v for k,v in env_score_dict.items() if k in grid_envs}


    ################################### Plot sorted trials and mean


    xlabel = 'Sorted by mean sample score'
    ylabel = 'Sample trial scores'
    plot_fname = 'trials_mean_sorted'

    plt.close('all')
    fig = plt.figure(figsize=(4,3))
    ax = plt.gca()
    scores = env_score_dict[solo_env]
    all_trials = sorted(scores, key=lambda x: np.mean(x))
    all_trials_mean = np.mean(all_trials, axis=1)

    ax.tick_params(**solo_plot_tick_params)

    ax.plot(all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
    ax.plot(all_trials_mean, color='black')

    ax.set_xlabel(xlabel, **solo_plot_label_params)
    ax.set_ylabel(ylabel, **solo_plot_label_params)

    ax.set_title(f'{solo_env} environment', **solo_plot_title_params)

    plt.tight_layout()
    plt.savefig(os.path.join(params_dir, f'{params_fname_label}_{plot_fname}_{solo_env}.png'))


    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(6,6))
    axes_flat = [axes[0][0], axes[1][0], axes[0][1], axes[1][1]]

    for i, (env, scores) in enumerate(env_score_dict_grid.items()):

        if scores is not None:
            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)

            axes_flat[i].tick_params(**plot_tick_params)

            axes_flat[i].plot(all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
            axes_flat[i].plot(all_trials_mean, color='black')

            axes_flat[i].set_xlabel(xlabel, **plot_label_params)
            axes_flat[i].set_ylabel(ylabel, **plot_label_params)

            axes_flat[i].set_title(f'{env} \nenvironment', **plot_title_params)

    plt.tight_layout()
    plt.savefig(os.path.join(params_dir, f'{params_fname_label}_{plot_fname}_2x2.png'))



    ################################### Plot sorted variances

    xlabel = 'Mean sample score'
    ylabel = 'Sample score variance'
    plot_fname = 'sample_trials_variance'


    plt.close('all')
    fig = plt.figure(figsize=(4,3))
    ax = plt.gca()
    scores = env_score_dict[solo_env]
    all_trials_sigma = np.std(scores, axis=1)
    all_trials_mean = np.mean(scores, axis=1)

    ax.tick_params(**solo_plot_tick_params)

    ax.plot(all_trials_mean, all_trials_sigma, 'o', color='dodgerblue', alpha=plot_pt_alpha, markersize=3)

    ax.set_xlabel(xlabel, **solo_plot_label_params)
    ax.set_ylabel(ylabel, **solo_plot_label_params)

    ax.set_title(f'{solo_env} environment', **solo_plot_title_params)

    plt.tight_layout()
    plt.savefig(os.path.join(params_dir, f'{params_fname_label}_{plot_fname}_{solo_env}.png'))


    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(6,6))
    axes_flat = [axes[0][0], axes[1][0], axes[0][1], axes[1][1]]

    for i, (env, scores) in enumerate(env_score_dict_grid.items()):
        if scores is not None:
            all_trials_sigma = np.std(scores, axis=1)
            all_trials_mean = np.mean(scores, axis=1)

            axes_flat[i].tick_params(**plot_tick_params)

            axes_flat[i].plot(all_trials_mean, all_trials_sigma, 'o', color='dodgerblue', alpha=plot_pt_alpha, markersize=3)

            axes_flat[i].set_xlabel(xlabel, **plot_label_params)
            axes_flat[i].set_ylabel(ylabel, **plot_label_params)

            axes_flat[i].set_title(f'{env} \nenvironment', **plot_title_params)

    plt.tight_layout()
    plt.savefig(os.path.join(params_dir, f'{params_fname_label}_{plot_fname}_2x2.png'))



    ################################### Plot log hists
    xlabel = 'Mean sample score'
    ylabel = 'log(counts)'
    plot_fname = 'mean_sample_score_log_hist'


    plt.close('all')
    fig = plt.figure(figsize=(4,3))
    ax = plt.gca()
    scores = env_score_dict[solo_env]
    all_trials_mean = np.mean(scores, axis=1)

    ax.tick_params(**solo_plot_tick_params)

    ax.hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True)

    ax.set_xlabel(xlabel, **solo_plot_label_params)
    ax.set_ylabel(ylabel, **solo_plot_label_params)

    ax.set_title(f'{solo_env} environment', **solo_plot_title_params)

    plt.tight_layout()
    plt.savefig(os.path.join(params_dir, f'{params_fname_label}_{plot_fname}_{solo_env}.png'))


    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(6,6))
    axes_flat = [axes[0][0], axes[1][0], axes[0][1], axes[1][1]]


    for i, (env, scores) in enumerate(env_score_dict_grid.items()):
        if scores is not None:
            all_trials_mean = np.mean(scores, axis=1)

            axes_flat[i].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True)

            axes_flat[i].tick_params(**plot_tick_params)

            axes_flat[i].set_xlabel(xlabel, **plot_label_params)
            axes_flat[i].set_ylabel(ylabel, **plot_label_params)

            axes_flat[i].set_title(f'{env} \nenvironment', **plot_title_params)

    plt.tight_layout()
    plt.savefig(os.path.join(params_dir, f'{params_fname_label}_{plot_fname}_2x2.png'))






def plot_envs_vs_NN_arch(stats_dir, **kwargs):
    '''
    For plotting a 5x5 grid of envs on one axis, and NN arch's used to
    solve them on the other axis.
    '''

    figures_base_dir = os.path.join(path_utils.get_output_dir(), 'figures')
    figures_dir = os.path.join(figures_base_dir, '5x5_plots')

    '''all_runs_dir = os.path.join(stats_dir, 'all_runs')
    # Load csv that holds the names of all the dirs
    stats_overview_fname = os.path.join(stats_dir, 'vary_params_stats.csv')
    overview_df = pd.read_csv(stats_overview_fname)
    drop_list = ['result_ID', 'run_plot_label', 'mu_all_scores', 'sigma_all_scores', 'all_scores_99_perc']
    overview_df = overview_df.drop(drop_list, axis=1)'''

    envs_list = [
        'MountainCarContinuous-v0',
        'CartPole-v0',
        'MountainCar-v0',
        'Pendulum-v0',
        'Acrobot-v1',
    ]

    env_name_title_dict = {
        'MountainCarContinuous-v0' : 'MountainCar-\nContinuous-v0',
        'CartPole-v0' : 'CartPole-v0',
        'Acrobot-v1' : 'Acrobot-v1',
        'MountainCar-v0' : 'MountainCar-v0',
        'Pendulum-v0' : 'Pendulum-v0'
    }

    '''{
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
        },
    ]

    params_dict_list = []
    for env in envs_list:
        for arch_dict in arch_dict_list:
            params_dict = arch_dict.copy()
            params_dict['env_name'] = env
            params_dict['NN'] = 'FFNN'
            params_dict_list.append(params_dict)


    walk_dict_list = walk_multi_dir(stats_dir, params_dict_list)

    env_arch_score_dict = {}
    for w_dict in walk_dict_list:

        env_arch_list = [w_dict['env_name'], w_dict['N_hidden_layers']]
        if 'N_hidden_units' in w_dict.keys():
            env_arch_list.append(w_dict['N_hidden_units'])

        env_arch_tuple = tuple(env_arch_list)
        print(env_arch_tuple)
        env_arch_score_dict[env_arch_tuple] = w_dict['all_trials']

    '''for i, env_name in enumerate(envs_list):
        for j, arch_dict in enumerate(arch_list):

            subset_df = overview_df[
                (overview_df['env_name'] == env_name) & \
                (overview_df['N_hidden_layers'] == arch_dict['N_hidden_layers']) & \
                (overview_df['N_hidden_units'] == arch_dict['N_hidden_units'])
                ]

            run_fname_label = subset_df['run_fname_label'].values[0]

            match_dirs = [x for x in os.listdir(all_runs_dir) if run_fname_label in x]
            assert len(match_dirs)==1, 'Must only have one dir matching label!'
            vary_dir = match_dirs[0]
            print(vary_dir)

            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(env_arch_tuple)

            for root, dirs, files in os.walk(os.path.join(all_runs_dir, vary_dir)):
                if 'sample_stats.json' in files:
                    with open(os.path.join(root, 'sample_stats.json'), 'r') as f:
                        sample_dict = json.load(f)

                    env_arch_score_dict[env_arch_tuple] = sample_dict['all_trials']'''


    unit_plot_w = 2.7
    unit_plot_h = 0.55*unit_plot_w

    grid_fig_size = (unit_plot_w*len(envs_list), unit_plot_h*len(arch_dict_list))
    plot_pt_alpha = 0.1
    plot_label_params = {
        'fontsize' : 15
    }
    plot_ylabel_params = {
        'fontsize' : 12
    }
    plot_tick_params = {
        'axis' : 'both',
        'labelsize' : 12
    }
    plot_title_params = {
        'fontsize' : 12
    }

    plot_score_trials = False
    plot_variances = False
    plot_hists = True
    plot_combo_plot_v1 = False
    plot_combo_plot_v2 = False
    plot_combo_plot_v3 = False

    ####################################### Score trials
    if plot_score_trials:
        print('\nPlotting all score trials...')
        plt.close('all')
        '''fig, axes = plt.subplots(len(envs_list), len(arch_dict_list), sharex='col', sharey='row',
                                    gridspec_kw={'hspace': .1, 'wspace': 0}, figsize=(10,8))'''
        fig, axes = plt.subplots(len(arch_dict_list), len(envs_list), sharex='col', sharey='col',
                                    gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=grid_fig_size)

        for j, env_name in enumerate(envs_list):
            for i, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting mean and trials of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)


                # For coloring by all episode scores
                all_trials_indexed = [[[i, y] for y in x] for i,x in enumerate(all_trials)]
                all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
                perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_indexed[:,1], perc_cutoff)

                all_trials_below = np.array([x for x in all_trials_indexed if x[1]<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials_indexed if x[1]>=perc_cutoff_val])
                axes[i][j].plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)

                # For uniform coloring
                #axes[i][j].plot(range(len(all_trials)), all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)

                # For coloring by mean score
                '''perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_mean, perc_cutoff)
                all_trials_below = np.array([x for x in all_trials if np.mean(x)<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials if np.mean(x)>=perc_cutoff_val])
                axes[i][j].plot(range(len(all_trials_below)), all_trials_below, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(range(len(all_trials_mean)-len(all_trials_above), len(all_trials_mean)), all_trials_above, 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)
                '''

                axes[i][j].plot(all_trials_mean, color='black')

                axes[i][j].tick_params(**plot_tick_params)
                '''axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
                axes[i][j].set_ylabel(env_name_title_dict[env_name], **plot_label_params)'''
                axes[i][j].set_ylabel(arch_dict_to_label(arch_dict), **plot_label_params)
                axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                axes[i][j].label_outer()

        #plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.97)
        print('Plotting png...')
        plt.savefig(os.path.join(figures_dir, f'5x5_trials_sorted.png'), dpi=300)
        #print('Plotting pdf...')
        #plt.savefig(os.path.join(figures_dir, f'5x5_trials_sorted.pdf'), dpi=300)



    ####################################### Variances
    if plot_variances:
        print('\nPlotting all variances...')
        plt.close('all')
        fig, axes = plt.subplots(len(envs_list), len(arch_dict_list), sharex=False, sharey='row',
                                    gridspec_kw={'hspace': .5, 'wspace': 0}, figsize=grid_fig_size)

        for i, env_name in enumerate(envs_list):
            for j, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting variance of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]
                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)
                all_trials_std = np.std(all_trials, axis=1)

                axes[i][j].tick_params(**plot_tick_params)

                axes[i][j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha, markersize=3)

                if i == len(envs_list)-1:
                    axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
                if j == 0:
                    axes[i][j].set_ylabel(env_name_title_dict[env_name], **plot_label_params)
                #axes[i][j].label_outer()

        plt.savefig(os.path.join(figures_dir, f'5x5_variance.png'))



    ####################################### Histograms
    if plot_hists:

        print('\nPlotting all histograms...')
        plt.close('all')
        fig, axes = plt.subplots(len(arch_dict_list), len(envs_list), sharex=False, sharey=False,
                                    gridspec_kw={'hspace': 0.25, 'wspace': 0.25}, figsize=grid_fig_size)

        '''for i, env_name in enumerate(envs_list):
        for j, arch_dict in enumerate(arch_dict_list):'''
        for j, env_name in enumerate(envs_list):
            for i, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting log hist of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)

                axes[i][j].tick_params(**plot_tick_params, pad=0)
                #axes[i][j].tick_params(labelsize=8)

                axes[i][j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True, bins=20)
                if i==len(arch_dict_list)-1:
                    axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                else:
                    axes[i][j].tick_params(axis='x', labelsize=0)


                if env_name == 'MountainCarContinuous-v0':
                    axes[i][j].set_xlim((-60, 110))
                if env_name == 'MountainCar-v0':
                    axes[i][j].set_xlim((-205, -120))
                if env_name == 'Pendulum-v0':
                    axes[i][j].set_xlim((-2000, -750))
                if env_name == 'Acrobot-v1':
                    axes[i][j].set_xlim((-500, -50))
                #if i == len(envs_list)-1:
                if j == 0:
                    #axes[i][j].set_ylabel('No bias,\n 2HL 4HU', **plot_label_params)
                    axes[i][j].set_ylabel(kwargs.get('ylabel_prefix', '') + arch_dict_to_label(arch_dict), **plot_ylabel_params)
                #axes[i][j].label_outer()

        plt.subplots_adjust(bottom=0.18, top=0.98, left=0.08, right=0.99)
        plt.savefig(os.path.join(figures_dir, f'5x5_hist_log.png'))





    ####################################### combo plot v1
    if plot_combo_plot_v1:
        print('\nPlotting big combo plot...')
        plt.close('all')
        '''fig, axes = plt.subplots(len(envs_list), len(arch_dict_list), sharex='col', sharey='row',
                                    gridspec_kw={'hspace': .1, 'wspace': 0}, figsize=(10,8))'''
        N_row = len(arch_dict_list)+2
        N_col = len(envs_list)
        fig, axes = plt.subplots(N_row, N_col, sharex=False, sharey=False,
                                    gridspec_kw={'hspace': .25, 'wspace': 0.3}, figsize=(10,8))

        plot_tick_params = {
            'axis' : 'both',
            'labelsize' : 7,
            'which' : 'major',
            'pad' : 0

        }

        ############################# plot main episode/mean plots
        for j, env_name in enumerate(envs_list):
            for i, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting mean and trials of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)


                # For coloring by all episode scores
                all_trials_indexed = [[[i, y] for y in x] for i,x in enumerate(all_trials)]
                all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
                perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_indexed[:,1], perc_cutoff)

                all_trials_below = np.array([x for x in all_trials_indexed if x[1]<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials_indexed if x[1]>=perc_cutoff_val])
                axes[i][j].plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)

                # For uniform coloring
                #axes[i][j].plot(range(len(all_trials)), all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)

                # For coloring by mean score
                '''perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_mean, perc_cutoff)
                all_trials_below = np.array([x for x in all_trials if np.mean(x)<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials if np.mean(x)>=perc_cutoff_val])
                axes[i][j].plot(range(len(all_trials_below)), all_trials_below, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(range(len(all_trials_mean)-len(all_trials_above), len(all_trials_mean)), all_trials_above, 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)
                '''

                axes[i][j].plot(all_trials_mean, color='black')

                axes[i][j].tick_params(**plot_tick_params)

                '''axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
                axes[i][j].set_ylabel(env_name_title_dict[env_name], **plot_label_params)'''
                if i==2:
                    #axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                    axes[i][j].set_xlabel('', **plot_label_params)
                if j == 0:
                    axes[i][j].set_ylabel('$S_{a,n,e}$ and $M_{a,n}$\n' + arch_dict_to_label(arch_dict), **plot_label_params)


        ######################### Plot variance for 2HL4HU
        i = len(arch_dict_list)
        for j, env_name in enumerate(envs_list):
            arch_dict = {
                'N_hidden_layers' : 2,
                'N_hidden_units' : 4
            }
            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]

            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)
            all_trials_std = np.std(all_trials, axis=1)

            axes[i][j].tick_params(**plot_tick_params)

            axes[i][j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha, markersize=3)

            #axes[i][j].set_xlabel('$M_{a,n}$', **plot_label_params)
            if j == 0:
                axes[i][j].set_ylabel('$V_{a,n}$\n' + arch_dict_to_label(arch_dict), **plot_label_params)

            axes[i][j].set_xlabel('', **plot_label_params)

            #axes[i][j].label_outer()
            '''if i == len(envs_list)-1:
                axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
            '''


        ##################################### Plot histograms for 2HL4HU
        i = len(arch_dict_list) + 1
        for j, env_name in enumerate(envs_list):
            arch_dict = {
                'N_hidden_layers' : 2,
                'N_hidden_units' : 4
            }
            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]

            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)

            axes[i][j].tick_params(**plot_tick_params)

            axes[i][j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True, bins=20)

            axes[i][j].set_xlabel('$M_{a,n}$\n' + env_name_title_dict[env_name], **plot_label_params)
            if j == 0:
                axes[i][j].set_ylabel('log(Counts)\n' + arch_dict_to_label(arch_dict), **plot_label_params)

            #axes[i][j].label_outer()


        #plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.97, left=0.09, right=0.99)
        print('Plotting png...')
        plt.savefig(os.path.join(figures_dir, f'combo_plot_v1.png'), dpi=300)



    ###################################### combo plot v2
    if plot_combo_plot_v2:
        print('\nPlotting big combo plot...')

        w_space = 0.3
        fig_width = 8
        adjust_kwargs = {
            'left' : 0.08,
            'right' : 0.98,
            'top' : 0.99
        }
        part_1_h = fig_width/2
        part_2_h = (2/3.0)*part_1_h*1.2
        plot_tick_params = {
            'axis' : 'both',
            'labelsize' : 6,
            'which' : 'major',
            'pad' : 0

        }

        ############################# First part
        plt.close('all')
        N_row = len(arch_dict_list)
        N_col = len(envs_list)
        fig, axes = plt.subplots(N_row, N_col, sharex='col', sharey='col',
                                    gridspec_kw={'hspace': 0, 'wspace': w_space}, figsize=(fig_width, part_1_h))


        ###################### plot main episode/mean plots
        for j, env_name in enumerate(envs_list):
            for i, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting mean and trials of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)


                # For coloring by all episode scores
                all_trials_indexed = [[[i, y] for y in x] for i,x in enumerate(all_trials)]
                all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
                perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_indexed[:,1], perc_cutoff)

                all_trials_below = np.array([x for x in all_trials_indexed if x[1]<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials_indexed if x[1]>=perc_cutoff_val])
                axes[i][j].plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)

                # For uniform coloring
                #axes[i][j].plot(range(len(all_trials)), all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)

                # For coloring by mean score
                '''perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_mean, perc_cutoff)
                all_trials_below = np.array([x for x in all_trials if np.mean(x)<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials if np.mean(x)>=perc_cutoff_val])
                axes[i][j].plot(range(len(all_trials_below)), all_trials_below, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(range(len(all_trials_mean)-len(all_trials_above), len(all_trials_mean)), all_trials_above, 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)
                '''

                axes[i][j].plot(all_trials_mean, color='black')

                axes[i][j].tick_params(**plot_tick_params)

                '''axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
                axes[i][j].set_ylabel(env_name_title_dict[env_name], **plot_label_params)'''
                if i==2:
                    #axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                    axes[i][j].set_xlabel('$R_a(n)$', **plot_label_params)
                if j == 0:
                    axes[i][j].set_ylabel('$S_{a,n,e}$ and $M_{a,n}$\n' + arch_dict_to_label(arch_dict), **plot_label_params)


        plt.subplots_adjust(bottom=0.1, **adjust_kwargs)
        print('Plotting part 1 png...')
        plt.savefig(os.path.join(figures_dir, f'combo_plot_v2_part1.png'), dpi=300)






        ################################ second part
        plt.close('all')
        N_row = 2
        N_col = len(envs_list)
        fig, axes = plt.subplots(N_row, N_col, sharex='col', sharey=False,
                                    gridspec_kw={'hspace': 0, 'wspace': w_space}, figsize=(fig_width, part_2_h))


        ######################### Plot variance for 2HL4HU
        i = 0
        for j, env_name in enumerate(envs_list):
            arch_dict = {
                'N_hidden_layers' : 2,
                'N_hidden_units' : 4
            }
            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]

            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)
            all_trials_std = np.std(all_trials, axis=1)

            axes[i][j].tick_params(**plot_tick_params)

            axes[i][j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha, markersize=3)

            #axes[i][j].set_xlabel('$M_{a,n}$', **plot_label_params)
            axes[i][j].set_xlabel('', **plot_label_params)
            if j == 0:
                axes[i][j].set_ylabel('$V_{a,n}$\n' + arch_dict_to_label(arch_dict), **plot_label_params)




        ##################################### Plot histograms for 2HL4HU
        i = 1
        for j, env_name in enumerate(envs_list):
            arch_dict = {
                'N_hidden_layers' : 2,
                'N_hidden_units' : 4
            }
            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]

            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)

            axes[i][j].tick_params(**plot_tick_params)

            axes[i][j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True, bins=20)

            axes[i][j].set_xlabel('$M_{a,n}$\n' + env_name_title_dict[env_name], **plot_label_params)
            if j == 0:
                axes[i][j].set_ylabel('log(Counts)\n' + arch_dict_to_label(arch_dict), **plot_label_params)



        #plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, **adjust_kwargs)
        print('Plotting part 2 png...')
        plt.savefig(os.path.join(figures_dir, f'combo_plot_v2_part2.png'), dpi=300)



    ###################################### combo plot v3
    if plot_combo_plot_v3:
        print('\nPlotting big combo plot...')

        w_space = 0
        fig_height = 8
        adjust_kwargs = {
            'bottom' : 0.12,
            'right' : 0.98,
            'top' : 0.99
        }
        part_1_w = fig_height/1.6
        part_2_w = (2/3.0)*part_1_w*1.0
        plot_tick_params = {
            'axis' : 'both',
            'labelsize' : 6,
            'which' : 'major',
            'pad' : 0

        }

        ############################# First part
        plt.close('all')
        N_row = len(envs_list)
        N_col = len(arch_dict_list)
        fig, axes = plt.subplots(N_row, N_col, sharex='all', sharey='row',
                                    gridspec_kw={'hspace': 0, 'wspace': w_space}, figsize=(part_1_w, fig_height))


        ###################### plot main episode/mean plots
        for i, env_name in enumerate(envs_list):
            for j, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting mean and trials of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)


                # For coloring by all episode scores
                all_trials_indexed = [[[i, y] for y in x] for i,x in enumerate(all_trials)]
                all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
                perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_indexed[:,1], perc_cutoff)

                all_trials_below = np.array([x for x in all_trials_indexed if x[1]<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials_indexed if x[1]>=perc_cutoff_val])
                axes[i][j].plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)

                # For uniform coloring
                #axes[i][j].plot(range(len(all_trials)), all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)


                axes[i][j].plot(all_trials_mean, color='black')

                axes[i][j].tick_params(**plot_tick_params)

                #arch_dict_to_label(arch_dict)
                if i==len(envs_list)-1:
                    #axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                    axes[i][j].set_xlabel('$R_a(n)$,\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)
                if j == 0:
                    axes[i][j].set_ylabel(env_name_title_dict[env_name] + '\n\n$S_{a,n,e}$ and $M_{a,n}$', **plot_label_params)


        plt.subplots_adjust(left=0.2, **adjust_kwargs)
        print('Plotting part 1 png...')
        plt.savefig(os.path.join(figures_dir, f'combo_plot_v3_part1.png'), dpi=300)



        ################################ second part
        plt.close('all')
        N_row = len(envs_list)
        N_col = 2
        fig, axes = plt.subplots(N_row, N_col, sharex=False, sharey=False,
                                    gridspec_kw={'hspace': 0, 'wspace': 0.23}, figsize=(part_2_w, fig_height))


        ######################### Plot variance for 2HL4HU
        j = 0
        for i, env_name in enumerate(envs_list):
            arch_dict = {
                'N_hidden_layers' : 2,
                'N_hidden_units' : 4
            }
            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]

            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)
            all_trials_std = np.std(all_trials, axis=1)

            axes[i][j].tick_params(**plot_tick_params)
            axes[i][j].tick_params(axis='x', labelsize=0)

            axes[i][j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha, markersize=3)

            #axes[i][j].set_xlabel('$M_{a,n}$', **plot_label_params)
            axes[i][j].set_ylabel('$V_{a,n}$', **plot_label_params)
            if i == len(envs_list)-1:
                axes[i][j].set_xlabel('$M_{a,n}$\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)

            #axes[i][j].label_outer()




        ##################################### Plot histograms for 2HL4HU
        j = 1
        for i, env_name in enumerate(envs_list):
            arch_dict = {
                'N_hidden_layers' : 2,
                'N_hidden_units' : 4
            }
            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]

            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)


            axes[i][j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True, bins=20)
            axes[i][j].tick_params(**plot_tick_params)
            axes[i][j].tick_params(axis='x', labelsize=0)

            axes[i][j].set_ylabel('', **plot_label_params)
            if i == len(envs_list)-1:
                axes[i][j].set_xlabel('$M_{a,n}$\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)

            #axes[i][j].label_outer()

        #plt.tight_layout()
        plt.subplots_adjust(left=0.14, **adjust_kwargs)
        print('Plotting part 2 png...')
        plt.savefig(os.path.join(figures_dir, f'combo_plot_v3_part2.png'), dpi=300)



def walk_multi_dir(multi_dir, params_dict_list):

    params_results_dict_list = []
    for params_dict in params_dict_list:

        for root, dirs, files in os.walk(multi_dir):
            if 'run_params.json' in files:
                with open(os.path.join(root, 'run_params.json'), 'r') as f:
                    run_params_dict = json.load(f)

                # Check if all the keys are in the json and they have
                # the right values
                all_keys_match = True
                for k,v in params_dict.items():
                    if k not in run_params_dict.keys():
                        all_keys_match = False
                        break

                    if run_params_dict[k] != v:
                        all_keys_match = False
                        break

                # If it's the right json, get the sample_stats.
                if all_keys_match:
                    #print(f'Matching dir: {root}')
                    with open(os.path.join(root, 'sample_stats.json'), 'r') as f:
                        sample_stats = json.load(f)

                    all_trials = np.array(sample_stats['all_trials'])
                    all_trials_mean = np.mean(all_trials, axis=1)
                    params_results = params_dict.copy()
                    params_results['best_score'] = np.max(all_trials_mean)
                    params_results['percentile_99.9'] = np.percentile(all_trials_mean, 99.9)
                    params_results['all_trials'] = all_trials
                    if 'total_runtime' in sample_stats.keys():
                        params_results['total_runtime'] = sample_stats['total_runtime']
                    params_results_dict_list.append(params_results)
                    break


    #pp.pprint(params_results_dict_list)
    return params_results_dict_list





################################# Helper functions

def convert_old_vary_params_json(vary_params_fname):
    '''
    This is because I used to have all_params.json be of the format:
    {
    "NN": "FFNN",
    "env_name": [
        "CartPole-v0",
        "Acrobot-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v0"
    ],
    "N_hidden_layers": [
        0,
        1
    ]
    }

    But it'd make life a lot easier if it were like:
    {
    'const_params':
        {
            "NN": "FFNN",
        }
    'vary_params':
        {
            "env_name": [
                "CartPole-v0",
                "Acrobot-v1",
                "MountainCar-v0",
                "MountainCarContinuous-v0",
                "Pendulum-v0"
            ],
            "N_hidden_layers": [
                0,
                1
            ]
        }
    }

    So this function takes a all_params.json file and converts it to the new
    version if it has to be.
    '''


    # Get vary_params to check
    with open(vary_params_fname, 'r') as f:
        vary_params_dict = json.load(f)

    # If for some reason it has const_params but not vary_params keys, something
    # else is wrong.
    if 'const_params' not in vary_params_dict.keys():

        assert 'vary_params' not in vary_params_dict.keys(), 'vary_params in keys(), should not be'

        print(f'\nall_params.json file {vary_params_fname} is of old format, reformatting...')

        combined_dict = {}
        combined_dict['const_params'] = {}
        combined_dict['vary_params'] = {}
        for k,v in vary_params_dict.items():

            # If the value is a list, that means it's a vary param, so
            # add it to that dict.
            if isinstance(v, list):
                combined_dict['vary_params'][k] = v
            else:
                combined_dict['const_params'][k] = v


        # Save combined params to file in new format
        with open(vary_params_fname, 'w+') as f:
            json.dump(combined_dict, f, indent=4)


def vary_params_dict_to_fname_label(vary_params_dict):
    return '_'.join([f'{k}={v}' for k,v in vary_params_dict.items()])


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



def replot_whole_stats_dir(stats_dir, **kwargs):

    if kwargs.get('replot_sample_dirs', False):
        print('\nReplotting all sample dirs...\n')

        for root, dirs, files in os.walk(stats_dir):
            if 'run_params.json' in files:
                print('Replotting for dir {}'.format(root.split('/')[-1]))
                replot_sample_dict_from_dir(root)


    if kwargs.get('replot_agg_stats', True):

        all_params_fname = os.path.join(stats_dir, 'all_params.json')
        assert os.path.exists(all_params_fname), f'all_params.json file {all_params_fname} DNE!'

        # fix if necessary
        convert_old_vary_params_json(all_params_fname)

        make_total_score_df(stats_dir)

        plot_all_agg_stats(stats_dir)



def arch_dict_to_label(arch_dict):

    label = '{} HL'.format(arch_dict['N_hidden_layers'])
    if 'N_hidden_units' in arch_dict.keys():
        label += ', {} HU'.format(arch_dict['N_hidden_units'])

    return label
#
