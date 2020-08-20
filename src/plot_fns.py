import numpy as np
import matplotlib.pyplot as plt
import os

PLOT_PT_ALPHA = 0.2
PLOT_LABEL_PARAMS = {"fontsize": 18}
PLOT_TICK_PARAMS = {"fontsize": 13}
PLOT_TITLE_PARAMS = {"fontsize": 18}


def plot_score_percentiles(save_dir, sample_dict, env_name, dt_str, **kwargs):

    plot_pt_alpha = kwargs.get("plot_pt_alpha", PLOT_PT_ALPHA)
    plot_label_params = kwargs.get("plot_label_params", PLOT_LABEL_PARAMS)
    plot_tick_params = kwargs.get("plot_tick_params", PLOT_TICK_PARAMS)
    plot_title_params = kwargs.get("plot_title_params", PLOT_TITLE_PARAMS)

    ###################### In mean order, with all trials

    all_trials = sample_dict["all_trials"]
    # all_trials_mean = np.mean(all_trials, axis=1)
    all_trials_mean = [np.mean(x) for x in all_trials]

    percs = np.linspace(0, 100.0, 20)
    perc_values = np.percentile(all_trials_mean, percs)

    percs_10 = np.arange(10, 100, 5).tolist() + [96, 97, 98, 99]
    perc_10_values = np.percentile(all_trials_mean, percs_10)

    plt.close("all")
    # plt.plot(all_trials, 'o', color='tomato', alpha=plot_pt_alpha)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for perc, val in zip(percs_10, perc_10_values):
        ax.axvline(
            perc,
            linestyle="dashed",
            color="gray",
            linewidth=0.5,
            label=f"perc {perc} = {val:.1f}",
        )
        ax.axhline(val, linestyle="dashed", color="gray", linewidth=0.5)

    plt.plot(percs, perc_values, color="mediumseagreen")

    plt.xlabel("Percentile", **plot_label_params)
    plt.ylabel("Percentile value", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(f"{env_name} environment", **plot_title_params)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir, "{}_percentiles_{}.png".format(env_name, dt_str),
        )
    )


def plot_scores(save_dir, sample_dict, env_name, dt_str, **kwargs):

    """
    For plotting results. Pass it a dict of the form
    returned by sample().

    Plots several versions of the same data (only mean, in the order they're
    run, mean, but ordered by increasing value, and then the mean and the scores
    for each trial).
    """

    plot_pt_alpha = kwargs.get("plot_pt_alpha", PLOT_PT_ALPHA)
    plot_label_params = kwargs.get("plot_label_params", PLOT_LABEL_PARAMS)
    plot_tick_params = kwargs.get("plot_tick_params", PLOT_TICK_PARAMS)
    plot_title_params = kwargs.get("plot_title_params", PLOT_TITLE_PARAMS)

    ###################### In time order

    plt.close("all")
    plt.plot(
        sample_dict["all_scores"], color="dodgerblue", label="All mean scores",
    )

    plt.xlabel("Sample", **plot_label_params)
    plt.ylabel("Sample mean score", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    # plt.legend()
    plt.title(f"{env_name} environment", **plot_title_params)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            "{}_score_mean_timeseries_{}.png".format(env_name, dt_str),
        )
    )

    ###################### In mean order

    all_scores = sample_dict["all_scores"]
    all_scores = sorted(all_scores)

    plt.close("all")
    plt.plot(all_scores, color="mediumseagreen")

    plt.xlabel("Sorted by sample mean score", **plot_label_params)
    plt.ylabel("Sample mean score", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    # plt.legend()
    plt.title(f"{env_name} environment", **plot_title_params)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir, "{}_score_mean_ordered_{}.png".format(env_name, dt_str),
        )
    )

    ###################### In mean order, with all trials

    all_trials = sample_dict["all_trials"]
    all_trials = sorted(all_trials, key=lambda x: np.mean(x))

    # all_trials_mean = np.mean(all_trials, axis=1)
    all_trials_mean = [np.mean(x) for x in all_trials]

    # For coloring by all episode scores
    all_trials_indexed = [
        [[i, y] for y in x] for i, x in enumerate(all_trials)
    ]

    # all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
    all_trials_indexed = np.array(sum(all_trials_indexed, []))

    perc_cutoff = 99.9
    perc_cutoff_val = np.percentile(all_trials_indexed[:, 1], perc_cutoff)

    all_trials_below = np.array(
        [x for x in all_trials_indexed if x[1] < perc_cutoff_val]
    )
    all_trials_above = np.array(
        [x for x in all_trials_indexed if x[1] >= perc_cutoff_val]
    )

    plt.close("all")
    plt.plot(
        *all_trials_below.transpose(),
        "o",
        color="tomato",
        alpha=plot_pt_alpha,
        markersize=3,
    )
    plt.plot(
        *all_trials_above.transpose(),
        "o",
        color="mediumseagreen",
        alpha=plot_pt_alpha,
        markersize=3,
    )
    plt.plot(all_trials_mean, color="black")

    plt.xlabel("Sorted by $R_a(n)$", **plot_label_params)
    plt.ylabel("$S_{a,n,e}$ and $M_{a,n}$", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    # plt.legend()
    plt.title(f"{env_name} environment", **plot_title_params)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            "{}_score_trials_ordered_{}.png".format(env_name, dt_str),
        )
    )


def plot_weight_stats(save_dir, sample_dict, env_name, dt_str, **kwargs):

    """
    For plotting episode mean scores and the corresponding L1 or L2 sums
    of the weight matrix that produced those scores.

    """

    plot_pt_alpha = kwargs.get("plot_pt_alpha", PLOT_PT_ALPHA)
    plot_label_params = kwargs.get("plot_label_params", PLOT_LABEL_PARAMS)
    plot_tick_params = kwargs.get("plot_tick_params", PLOT_TICK_PARAMS)
    plot_title_params = kwargs.get("plot_title_params", PLOT_TITLE_PARAMS)

    L0_weights = sample_dict["L0_weights"]
    L1_weights = sample_dict["L1_weights"]
    L2_weights = sample_dict["L2_weights"]
    all_scores = sample_dict["all_scores"]

    ###################### L0
    plt.close("all")
    plt.plot(
        all_scores, L0_weights, "o", color="forestgreen", alpha=plot_pt_alpha,
    )

    plt.xlabel("Sample mean score", **plot_label_params)
    plt.ylabel("L0/N_weights", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    # plt.legend()
    plt.title(
        f"{env_name} environment,\n L0 sum of weights", **plot_title_params,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir, "{}_L0_vs_meanscore_{}.png".format(env_name, dt_str),
        )
    )

    ###################### L1
    plt.close("all")
    plt.plot(
        all_scores, L1_weights, "o", color="forestgreen", alpha=plot_pt_alpha,
    )

    plt.xlabel("Sample mean score", **plot_label_params)
    plt.ylabel("L1/N_weights", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    # plt.legend()
    plt.title(
        f"{env_name} environment,\n L1 sum of weights", **plot_title_params,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir, "{}_L1_vs_meanscore_{}.png".format(env_name, dt_str),
        )
    )

    ######################## L2
    plt.close("all")
    plt.plot(
        all_scores, L2_weights, "o", color="forestgreen", alpha=plot_pt_alpha,
    )

    plt.xlabel("Sample mean score", **plot_label_params)
    plt.ylabel("L2/N_weights", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    # plt.legend()
    plt.title(
        f"{env_name} environment,\n L2 sum of weights", **plot_title_params,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir, "{}_L2_vs_meanscore_{}.png".format(env_name, dt_str),
        )
    )


def plot_all_trial_stats(save_dir, sample_dict, env_name, dt_str, **kwargs):

    """
    Plots the variance, min, and max of the scores for the N_episodes of
    each episode, as a function of the mean score for that episode.

    """

    plot_pt_alpha = kwargs.get("plot_pt_alpha", PLOT_PT_ALPHA)
    plot_label_params = kwargs.get("plot_label_params", PLOT_LABEL_PARAMS)
    plot_tick_params = kwargs.get("plot_tick_params", PLOT_TICK_PARAMS)
    plot_title_params = kwargs.get("plot_title_params", PLOT_TITLE_PARAMS)

    N_samples = len(sample_dict["all_trials"])
    N_episodes = len(sample_dict["all_trials"][0])

    ####################### Episode score variance
    plt.close("all")

    # sigma = np.std(sample_dict["all_trials"], axis=1)
    sigma = [np.std(x) for x in sample_dict["all_trials"]]

    plt.plot(
        sample_dict["all_scores"],
        sigma,
        "o",
        color="mediumorchid",
        alpha=plot_pt_alpha,
    )

    plt.xlabel("$M_a(n)$", **plot_label_params)
    plt.ylabel("$V_{a,n}$", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    plt.title(f"{env_name} environment", **plot_title_params)
    plt.tight_layout()
    fname = os.path.join(
        save_dir, "{}_variance_meanscore_{}.png".format(env_name, dt_str),
    )
    plt.savefig(fname)

    ####################### Min sample score
    plt.close("all")

    # trial_min = np.min(sample_dict["all_trials"], axis=1)
    trial_min = [np.min(x) for x in sample_dict["all_trials"]]

    plt.plot(
        sample_dict["all_scores"],
        trial_min,
        "o",
        color="dodgerblue",
        alpha=plot_pt_alpha,
    )

    plt.xlabel("Sample mean score", **plot_label_params)
    plt.ylabel("Min of sample scores", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    plt.title(
        f"{env_name} environment,\n min score of N_episodes = {N_episodes}",
        **plot_title_params,
    )
    plt.tight_layout()
    fname = os.path.join(
        save_dir, "{}_min_score_{}.png".format(env_name, dt_str),
    )
    plt.savefig(fname)

    ####################### Max episode score
    plt.close("all")

    # trial_max = np.max(sample_dict["all_trials"], axis=1)
    trial_max = [np.max(x) for x in sample_dict["all_trials"]]

    plt.plot(
        sample_dict["all_scores"],
        trial_max,
        "o",
        color="dodgerblue",
        alpha=plot_pt_alpha,
    )

    plt.xlabel("Sample mean score", **plot_label_params)
    plt.ylabel("Max of sample scores", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    plt.title(
        f"{env_name} environment,\n max score of N_episodes = {N_episodes}",
        **plot_title_params,
    )
    plt.tight_layout()
    fname = os.path.join(
        save_dir, "{}_max_score_{}.png".format(env_name, dt_str),
    )
    plt.savefig(fname)

    ####################### Min and max episode score
    plt.close("all")

    # trial_min = np.min(sample_dict["all_trials"], axis=1)
    # trial_max = np.max(sample_dict["all_trials"], axis=1)
    trial_min = [np.min(x) for x in sample_dict["all_trials"]]
    trial_max = [np.max(x) for x in sample_dict["all_trials"]]

    plt.plot(
        sample_dict["all_scores"],
        trial_min,
        "o",
        color="mediumturquoise",
        alpha=plot_pt_alpha,
    )
    plt.plot(
        sample_dict["all_scores"],
        trial_max,
        "o",
        color="plum",
        alpha=plot_pt_alpha,
    )

    plt.xlabel("Sample mean score", **plot_label_params)
    plt.ylabel("Min and max of sample scores", **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    plt.title(
        f"{env_name} environment, min (turquoise) and \nmax (purple) score of N_episodes = {N_episodes}",
        **plot_title_params,
    )
    plt.tight_layout()
    fname = os.path.join(
        save_dir, "{}_min_max_score_{}.png".format(env_name, dt_str),
    )
    plt.savefig(fname)


def plot_sample_histogram(
    save_dir, dist, dist_label, fname, env_name, dt_str, **kwargs
):

    """
    For plotting the distribution of various benchmarking stats for env_name.
    Plots a vertical dashed line at the mean.

    kwarg plot_log = True also plots one with a log y axis, which is often
    better because the number of best solutions are very small.
    """

    plot_pt_alpha = kwargs.get("plot_pt_alpha", PLOT_PT_ALPHA)
    plot_label_params = kwargs.get("plot_label_params", PLOT_LABEL_PARAMS)
    plot_tick_params = kwargs.get("plot_tick_params", PLOT_TICK_PARAMS)
    plot_title_params = kwargs.get("plot_title_params", PLOT_TITLE_PARAMS)

    fname = os.path.join(save_dir, fname)

    plt.close("all")
    # mu = np.mean(dist)
    # sd = np.std(dist)

    if kwargs.get("N_bins", None) is None:
        plt.hist(dist, color="dodgerblue", edgecolor="gray")
    else:
        plt.hist(
            dist,
            color="dodgerblue",
            edgecolor="gray",
            bins=kwargs.get("N_bins", None),
        )

    # plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
    # plt.xlabel(dist_label, **plot_label_params)

    plt.xlabel("$M_{a,n}$", **plot_label_params)
    plt.ylabel("Counts", **plot_label_params)
    # plt.ylabel('$S_{t,n,r}$ and $M_{t,n}$', **plot_label_params)

    plt.xticks(**plot_tick_params)
    plt.yticks(**plot_tick_params)

    # plt.title(f'{dist_label} distribution for {env_name}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$', **plot_title_params)
    # plt.title(f'{dist_label} distribution \nfor {env_name}', **plot_title_params)
    plt.title(f"{env_name} environment", **plot_title_params)
    plt.savefig(fname)

    if kwargs.get("plot_log", False):
        if kwargs.get("N_bins", None) is None:
            plt.hist(dist, color="dodgerblue", edgecolor="gray", log=True)
        else:
            plt.hist(
                dist,
                color="dodgerblue",
                edgecolor="gray",
                bins=kwargs.get("N_bins", None),
                log=True,
            )

        # plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
        # plt.xlabel(dist_label, **plot_label_params)
        plt.xlabel("$M_{a,n}$", **plot_label_params)
        plt.ylabel("log(Counts)", **plot_label_params)

        plt.xticks(**plot_tick_params)
        plt.yticks(**plot_tick_params)

        # plt.title(f'{dist_label} distribution for {env_name}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$', **plot_title_params)
        # plt.title(f'{dist_label} distribution \nfor {env_name}', **plot_title_params)
        plt.title(f"{env_name} environment", **plot_title_params)
        plt.tight_layout()
        plt.savefig(fname.replace("dist", "log_dist"))
