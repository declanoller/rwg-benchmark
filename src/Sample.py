import path_utils
import numpy as np
import gym
import matplotlib.pyplot as plt
import os, json, time

import Agent

gym.logger.set_level(40)

"""

Sample class
--------------------

Used to run an agent in a gym env over a number (N_samples) of samples. Each
sample, the same agent is run for N_episodes, because for many envs, different
initial conditions will give different scores for the same agent.

After every sample, Sample.get_next_sample() is called, which gets
the next agent. For RWG, this simply involves picking a new set of random
weights, so there's no real "progression" or relation between samples.
However, this offers the opportunity to use other methods (CMA-ES, etc).

Uses the Agent class, which handles the NN and related stuff.

"""


class Sample:
    def __init__(self, env_name, **kwargs):

        # The search method used. Default is Random Weight Guessing (RWG).
        self.search_method = kwargs.get("search_method", "RWG")
        assert self.search_method in [
            "RWG",
            "gaussian_noise_hill_climb",
            "grid_search",
            "bin_grid_search",
            "sparse_bin_grid_search",
        ], "Must supply valid search_method!"

        # Create env, create agent
        self.setup_env(env_name)
        self.agent = Agent.Agent(self.env, **kwargs)

        self.noise_sd = 1.0
        self.max_episode_steps = kwargs.get("max_episode_steps", 500)

        # Get the base dir, which is where runs will be saved to. Default
        # is /output/
        base_dir = kwargs.get("base_dir", path_utils.get_output_dir())

        # Datetime string for labeling the run
        self.dt_str = path_utils.get_date_str()

        # If you don't pass anything, it will create a dir in base_dir to
        # hold the results of this run, but you can supply your own externally.
        self.run_dir = kwargs.get("run_dir", None)
        if self.run_dir is None:
            self.run_dir = os.path.join(
                base_dir, f"{self.env_name}_sample_{self.dt_str}"
            )
            os.mkdir(self.run_dir)

        # For saving the parameters used for the run. Run last in __init__().
        if kwargs.get("load_params_from_dir", False):
            self.load_params_dict()
        else:
            self.run_params = kwargs.copy()
            self.save_params_dict()

        #### Plot params
        self.plot_pt_alpha = 0.2
        self.plot_label_params = {"fontsize": 18}
        self.plot_tick_params = {"fontsize": 13}
        self.plot_title_params = {"fontsize": 18}

    def setup_env(self, env_name):

        """
        For setting up the env and getting the info
        about when it's solved, etc.
        """

        with open(
            os.path.join(path_utils.get_src_dir(), "gym_envs_info.json"), "r"
        ) as f:
            envs_dict = json.load(f)

        assert (
            env_name in envs_dict.keys()
        ), f"Env {env_name} not in envs_dict!"

        self.env_name = env_name
        self.env = gym.make(env_name)

        # Two details for being considered "solved"
        self.solved_avg_reward = envs_dict[env_name]["solved_avg_reward"]
        self.N_eval_trials = envs_dict[env_name]["N_eval_trials"]

    def sample(self, N_samples, **kwargs):

        """
        Sample the agent for N_samples samples,
        improving it with the selection mechanism.

        Each sample, use the same agent for N_episodes to try and get a more
        representative score from it.
        """

        N_episodes = kwargs.get("N_episodes", 3)

        all_scores = []
        best_scores = []
        all_trials = []
        L0_weights = []
        L1_weights = []
        L2_weights = []
        all_weights = []
        best_score = None
        best_weights = self.agent.get_weight_matrix()
        start_time = time.time()

        # Gameplay loop
        for samp_num in range(N_samples):

            if kwargs.get("print_samp_num", False):
                if samp_num % max(1, N_samples // 10) == 0:
                    print(f"Sample {samp_num}/{N_samples}")

            score_trials = []
            for _ in range(N_episodes):
                # Run episode, get score, record score
                score_trials.append(self.run_episode())

            # Take mean score of N_episodes, record if best score yet
            mean_score = np.mean(score_trials)
            if (best_score is None) or (mean_score > best_score):
                best_score = mean_score
                # print(f'New best score {best_score:.3f} in sample {samp_num}')
                best_weights = self.agent.get_weight_matrix()

            # Get stats about the weights of the NN
            weight_sum_dict = self.agent.get_weight_sums()
            L0_weights.append(weight_sum_dict["L0"])
            L1_weights.append(weight_sum_dict["L1"])
            L2_weights.append(weight_sum_dict["L2"])

            all_scores.append(mean_score)
            best_scores.append(best_score)
            all_trials.append(score_trials)

            # Get next agent.
            self.get_next_sample(all_scores, best_scores, best_weights)

            if self.agent.search_done:
                print(f"Search done in samp_num {samp_num}\n\n")
                break

        total_runtime = time.time() - start_time

        ret_dict = {
            "best_scores": best_scores,
            "all_scores": all_scores,
            "all_trials": all_trials,
            "best_weights": [bw.tolist() for bw in best_weights],
            "L0_weights": L0_weights,
            "L1_weights": L1_weights,
            "L2_weights": L2_weights,
            "N_samples": N_samples,
            "N_episodes": N_episodes,
            "total_runtime": total_runtime,
            "best_score": best_score,
        }

        if kwargs.get("save_all_weights", False):
            ret_dict["all_weights"] = all_weights

        return ret_dict

    def run_episode(self, **kwargs):

        """
        Run episode with gym env. Returns the total score
        for the episode. Pass show_ep=True to render the episode.
        """

        show_ep = kwargs.get("show_ep", False)

        # For recording a movie of the agent.
        if kwargs.get("record_ep", False):
            self.env = gym.wrappers.Monitor(self.env, self.run_dir, force=True)

        obs = self.env.reset()
        self.agent.init_episode()
        score = 0
        steps = 0
        done = False
        while not done:
            if show_ep:
                self.env.render()
                if steps % 10 == 0:
                    print(f"step {steps}, score {score:.2f}")

            action = self.agent.get_action(obs)
            obs, rew, done, info = self.env.step(action)
            score += rew
            steps += 1
            if steps >= self.max_episode_steps:
                done = True

        if show_ep:
            self.env.close()
            print(f"Score = {score:.3f}")

        return score

    def get_next_sample(self, all_scores, best_scores, best_weights):

        """
        For getting the next sample using some selection criteria.
        Right now it's just RWG, which resets the weights randomly.
        """

        if self.search_method == "RWG":
            self.agent.set_random_weights()

        elif self.search_method in [
            "grid_search",
            "bin_grid_search",
            "sparse_bin_grid_search",
        ]:

            self.agent.mutate_grid_search()

    ################################ Plotting/saving functions

    def plot_score_percentiles(self, sample_dict, **kwargs):

        ###################### In mean order, with all trials

        all_trials = sample_dict["all_trials"]
        all_trials_mean = np.mean(all_trials, axis=1)

        percs = np.linspace(0, 100.0, 20)
        perc_values = np.percentile(all_trials_mean, percs)

        percs_10 = np.arange(10, 100, 5).tolist() + [96, 97, 98, 99]
        perc_10_values = np.percentile(all_trials_mean, percs_10)

        plt.close("all")
        # plt.plot(all_trials, 'o', color='tomato', alpha=self.plot_pt_alpha)
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

        plt.xlabel("Percentile", **self.plot_label_params)
        plt.ylabel("Percentile value", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"{self.env_name} environment", **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.run_dir,
                "{}_percentiles_{}.png".format(self.env_name, self.dt_str),
            )
        )

    def plot_scores(self, sample_dict, **kwargs):

        """
        For plotting results. Pass it a dict of the form
        returned by sample().

        Plots several versions of the same data (only mean, in the order they're
        run, mean, but ordered by increasing value, and then the mean and the scores
        for each trial).
        """

        ###################### In time order

        plt.close("all")
        plt.plot(
            sample_dict["all_scores"],
            color="dodgerblue",
            label="All mean scores",
        )

        plt.xlabel("Sample", **self.plot_label_params)
        plt.ylabel("Sample mean score", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f"{self.env_name} environment", **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.run_dir,
                "{}_score_mean_timeseries_{}.png".format(
                    self.env_name, self.dt_str
                ),
            )
        )

        ###################### In mean order

        all_scores = sample_dict["all_scores"]
        all_scores = sorted(all_scores)

        plt.close("all")
        plt.plot(all_scores, color="mediumseagreen")

        plt.xlabel("Sorted by sample mean score", **self.plot_label_params)
        plt.ylabel("Sample mean score", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f"{self.env_name} environment", **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.run_dir,
                "{}_score_mean_ordered_{}.png".format(
                    self.env_name, self.dt_str
                ),
            )
        )

        ###################### In mean order, with all trials

        all_trials = sample_dict["all_trials"]
        all_trials = sorted(all_trials, key=lambda x: np.mean(x))

        all_trials_mean = np.mean(all_trials, axis=1)

        # For coloring by all episode scores
        all_trials_indexed = [
            [[i, y] for y in x] for i, x in enumerate(all_trials)
        ]
        all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))

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
            alpha=self.plot_pt_alpha,
            markersize=3,
        )
        plt.plot(
            *all_trials_above.transpose(),
            "o",
            color="mediumseagreen",
            alpha=self.plot_pt_alpha,
            markersize=3,
        )
        plt.plot(all_trials_mean, color="black")

        plt.xlabel("Sorted by $R_a(n)$", **self.plot_label_params)
        plt.ylabel("$S_{a,n,e}$ and $M_{a,n}$", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f"{self.env_name} environment", **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.run_dir,
                "{}_score_trials_ordered_{}.png".format(
                    self.env_name, self.dt_str
                ),
            )
        )

    def plot_weight_stats(self, sample_dict, **kwargs):

        """
        For plotting episode mean scores and the corresponding L1 or L2 sums
        of the weight matrix that produced those scores.

        """

        L0_weights = sample_dict["L0_weights"]
        L1_weights = sample_dict["L1_weights"]
        L2_weights = sample_dict["L2_weights"]
        all_scores = sample_dict["all_scores"]

        ###################### L0
        plt.close("all")
        plt.plot(
            all_scores,
            L0_weights,
            "o",
            color="forestgreen",
            alpha=self.plot_pt_alpha,
        )

        plt.xlabel("Sample mean score", **self.plot_label_params)
        plt.ylabel("L0/N_weights", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(
            f"{self.env_name} environment,\n L0 sum of weights",
            **self.plot_title_params,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.run_dir,
                "{}_L0_vs_meanscore_{}.png".format(self.env_name, self.dt_str),
            )
        )

        ###################### L1
        plt.close("all")
        plt.plot(
            all_scores,
            L1_weights,
            "o",
            color="forestgreen",
            alpha=self.plot_pt_alpha,
        )

        plt.xlabel("Sample mean score", **self.plot_label_params)
        plt.ylabel("L1/N_weights", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(
            f"{self.env_name} environment,\n L1 sum of weights",
            **self.plot_title_params,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.run_dir,
                "{}_L1_vs_meanscore_{}.png".format(self.env_name, self.dt_str),
            )
        )

        ######################## L2
        plt.close("all")
        plt.plot(
            all_scores,
            L2_weights,
            "o",
            color="forestgreen",
            alpha=self.plot_pt_alpha,
        )

        plt.xlabel("Sample mean score", **self.plot_label_params)
        plt.ylabel("L2/N_weights", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(
            f"{self.env_name} environment,\n L2 sum of weights",
            **self.plot_title_params,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.run_dir,
                "{}_L2_vs_meanscore_{}.png".format(self.env_name, self.dt_str),
            )
        )

    def plot_all_trial_stats(self, sample_dict, **kwargs):

        """
        Plots the variance, min, and max of the scores for the N_episodes of
        each episode, as a function of the mean score for that episode.

        """

        N_samples = len(sample_dict["all_trials"])
        N_episodes = len(sample_dict["all_trials"][0])

        ####################### Episode score variance
        plt.close("all")

        sigma = np.std(sample_dict["all_trials"], axis=1)

        plt.plot(
            sample_dict["all_scores"],
            sigma,
            "o",
            color="mediumorchid",
            alpha=self.plot_pt_alpha,
        )

        plt.xlabel("$M_a(n)$", **self.plot_label_params)
        plt.ylabel("$V_{a,n}$", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f"{self.env_name} environment", **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(
            self.run_dir,
            "{}_variance_meanscore_{}.png".format(self.env_name, self.dt_str),
        )
        plt.savefig(fname)

        ####################### Min sample score
        plt.close("all")

        trial_min = np.min(sample_dict["all_trials"], axis=1)

        plt.plot(
            sample_dict["all_scores"],
            trial_min,
            "o",
            color="dodgerblue",
            alpha=self.plot_pt_alpha,
        )

        plt.xlabel("Sample mean score", **self.plot_label_params)
        plt.ylabel("Min of sample scores", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(
            f"{self.env_name} environment,\n min score of N_episodes = {N_episodes}",
            **self.plot_title_params,
        )
        plt.tight_layout()
        fname = os.path.join(
            self.run_dir,
            "{}_min_score_{}.png".format(self.env_name, self.dt_str),
        )
        plt.savefig(fname)

        ####################### Max episode score
        plt.close("all")

        trial_max = np.max(sample_dict["all_trials"], axis=1)

        plt.plot(
            sample_dict["all_scores"],
            trial_max,
            "o",
            color="dodgerblue",
            alpha=self.plot_pt_alpha,
        )

        plt.xlabel("Sample mean score", **self.plot_label_params)
        plt.ylabel("Max of sample scores", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(
            f"{self.env_name} environment,\n max score of N_episodes = {N_episodes}",
            **self.plot_title_params,
        )
        plt.tight_layout()
        fname = os.path.join(
            self.run_dir,
            "{}_max_score_{}.png".format(self.env_name, self.dt_str),
        )
        plt.savefig(fname)

        ####################### Min and max episode score
        plt.close("all")

        trial_min = np.min(sample_dict["all_trials"], axis=1)
        trial_max = np.max(sample_dict["all_trials"], axis=1)

        plt.plot(
            sample_dict["all_scores"],
            trial_min,
            "o",
            color="mediumturquoise",
            alpha=self.plot_pt_alpha,
        )
        plt.plot(
            sample_dict["all_scores"],
            trial_max,
            "o",
            color="plum",
            alpha=self.plot_pt_alpha,
        )

        plt.xlabel("Sample mean score", **self.plot_label_params)
        plt.ylabel("Min and max of sample scores", **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(
            f"{self.env_name} environment, min (turquoise) and \nmax (purple) score of N_episodes = {N_episodes}",
            **self.plot_title_params,
        )
        plt.tight_layout()
        fname = os.path.join(
            self.run_dir,
            "{}_min_max_score_{}.png".format(self.env_name, self.dt_str),
        )
        plt.savefig(fname)

    def show_best_episode(self, weights):

        """
        Pass it the weights matrix you want to run it with,
        e.g., best_weights returned from sample(). It runs
        an episode and renders it.
        """

        self.agent.set_weight_matrix(weights)
        ep_score = self.run_episode(show_ep=True, record_ep=False)

    def record_best_episode(self, weights):

        """
        Pass it the weights matrix you want to run it with,
        e.g., best_weights returned from sample(). It runs
        an episode and renders it.
        """

        self.agent.set_weight_matrix(weights)
        ep_score = self.run_episode(show_ep=True, record_ep=True)

    def save_all_sample_stats(self, sample_dict, **kwargs):
        """
        For saving all the stats and plots for the sampling, just a collector
        function.

        """

        self.save_sample_dict(sample_dict)
        if kwargs.get("save_plots", True):
            self.plot_scores(sample_dict)
            self.plot_all_trial_stats(sample_dict)
            self.plot_sample_histogram(
                sample_dict["all_scores"],
                "Mean sample score",
                f"{self.env_name}_all_scores_dist_{self.dt_str}.png",
                plot_log=True,
                **kwargs,
            )
            self.plot_weight_stats(sample_dict)
            self.plot_score_percentiles(sample_dict)

    def save_sample_dict(self, sample_dict):
        """
        For saving the results of the run in a .json file, for later analysis.
        """

        # Maybe not necessary, but to be careful to not modify the original
        sample_dict_copy = sample_dict.copy()
        """if "best_weights" in sample_dict_copy.keys():
            sample_dict_copy.pop("best_weights")"""

        fname = os.path.join(self.run_dir, "sample_stats.json")
        # Save distributions to file

        with open(fname, "w+") as f:
            json.dump(sample_dict_copy, f, indent=4)

    def save_params_dict(self):
        """
        For saving the parameters used in a .json file, for later analysis.
        """

        self.run_params["env_name"] = self.env_name
        self.run_params["search_method"] = self.search_method
        self.run_params["dt_str"] = self.dt_str
        self.run_params["run_dir"] = self.run_dir
        self.run_params["NN_type"] = self.agent.NN_type

        fname = os.path.join(self.run_dir, "run_params.json")
        # Save distributions to file

        with open(fname, "w+") as f:
            json.dump(self.run_params, f, indent=4)

    def load_params_dict(self):
        """
        For loading the parameters from a saved .json file, for later analysis.
        """
        fname = os.path.join(self.run_dir, "run_params.json")
        # Save distributions to file

        with open(fname, "r") as f:
            self.run_params = json.load(f)

        self.env_name = self.run_params["env_name"]
        self.search_method = self.run_params["search_method"]
        self.dt_str = self.run_params["dt_str"]
        self.run_dir = self.run_params["run_dir"]
        self.agent.NN_type = self.run_params[
            "NN_type"
        ]  # not necessary probably? Be careful

    def plot_sample_histogram(self, dist, dist_label, fname, **kwargs):

        """
        For plotting the distribution of various benchmarking stats for self.env_name.
        Plots a vertical dashed line at the mean.

        kwarg plot_log = True also plots one with a log y axis, which is often
        better because the number of best solutions are very small.
        """

        fname = os.path.join(self.run_dir, fname)

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
        # plt.xlabel(dist_label, **self.plot_label_params)

        plt.xlabel("$M_{a,n}$", **self.plot_label_params)
        plt.ylabel("Counts", **self.plot_label_params)
        # plt.ylabel('$S_{t,n,r}$ and $M_{t,n}$', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.title(f'{dist_label} distribution for {self.env_name}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$', **self.plot_title_params)
        # plt.title(f'{dist_label} distribution \nfor {self.env_name}', **self.plot_title_params)
        plt.title(f"{self.env_name} environment", **self.plot_title_params)
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
            # plt.xlabel(dist_label, **self.plot_label_params)
            plt.xlabel("$M_{a,n}$", **self.plot_label_params)
            plt.ylabel("log(Counts)", **self.plot_label_params)

            plt.xticks(**self.plot_tick_params)
            plt.yticks(**self.plot_tick_params)

            # plt.title(f'{dist_label} distribution for {self.env_name}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$', **self.plot_title_params)
            # plt.title(f'{dist_label} distribution \nfor {self.env_name}', **self.plot_title_params)
            plt.title(f"{self.env_name} environment", **self.plot_title_params)
            plt.tight_layout()
            plt.savefig(fname.replace("dist", "log_dist"))


def replot_sample_dict_from_dir(dir, **kwargs):

    """
    Minor fix: originally this would open run_params.json and read the run_dir
    field, and pass that to the Sample() object. However, that caused trouble
    if the run was done on another machine, because the path was absolute,
    so it would then be looking for a path that might not exist on the machine
    that this function is being run on.

    Instead, since we're already assuming this dir is a run dir, it should just
    take this dir and rewrite run_dir.

    """

    assert os.path.exists(dir), f"Dir must exist to load from! Dir {dir} DNE."

    run_params_json_fname = os.path.join(dir, "run_params.json")
    assert os.path.exists(
        run_params_json_fname
    ), f"run_params.json must exist in dir to load from! {run_params_json_fname} DNE."

    sample_dict_fname = os.path.join(dir, "sample_stats.json")
    assert os.path.exists(
        sample_dict_fname
    ), f"sample_stats.json must exist in dir to load from! {sample_dict_fname} DNE."

    # Get run_params to recreate the object
    with open(run_params_json_fname, "r") as f:
        run_params = json.load(f)

    # Rewrite run_params in case it was originally run on another machine.
    run_params["run_dir"] = dir

    with open(run_params_json_fname, "w+") as f:
        json.dump(run_params, f, indent=4)

    # Recreate Sample object, get sample_dict, replot
    # Have to pass run_dir so it doesn't automatically create a new dir.
    e = Sample(
        run_params["env_name"],
        run_dir=run_params["run_dir"],
        load_params_from_dir=True,
    )

    # Get sample_dict to replot statistics found
    with open(sample_dict_fname, "r") as f:
        sample_dict = json.load(f)

    # Replot
    e.save_all_sample_stats(sample_dict, **kwargs)


def load_best_agent_sample_from_dir(dir):
    assert os.path.exists(dir), f"Dir must exist to load from! Dir {dir} DNE."

    run_params_json_fname = os.path.join(dir, "run_params.json")
    assert os.path.exists(
        run_params_json_fname
    ), f"run_params.json must exist in dir to load from! {run_params_json_fname} DNE."

    sample_dict_fname = os.path.join(dir, "sample_stats.json")
    assert os.path.exists(
        sample_dict_fname
    ), f"sample_stats.json must exist in dir to load from! {sample_dict_fname} DNE."

    # Get run_params to recreate the object
    with open(run_params_json_fname, "r") as f:
        run_params = json.load(f)

    del run_params["run_dir"]
    del run_params["dt_str"]
    print("\nPassing this dict to create a new Sample():")
    print(run_params)
    print()
    env_name = run_params.pop("env_name")

    s = Sample(env_name, **run_params)

    with open(sample_dict_fname, "r") as f:
        sample_dict = json.load(f)

    print(sample_dict.keys())
    best_weights = sample_dict["best_weights"]
    print("\nLoading best weights:")
    print(best_weights)
    best_weights_mat = [np.array(w) for w in best_weights]
    s.agent.set_weight_matrix(best_weights_mat)

    return s


#
