import path_utils
import gym
import numpy as np
import os, json, time

import Agent
import plot_fns

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

        self.plot_params = {
            "plot_pt_alpha": self.plot_pt_alpha,
            "plot_label_params": self.plot_label_params,
            "plot_tick_params": self.plot_tick_params,
            "plot_title_params": self.plot_title_params,
        }

    def setup_env(self, env_name):

        """
        For setting up the env.
        """

        self.env_name = env_name
        self.env = gym.make(env_name)

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
        try:
            for samp_num in range(N_samples):

                if kwargs.get("print_samp_num", False):
                    if samp_num % max(1, N_samples // 10) == 0:
                        print(f"Sample {samp_num}/{N_samples}")

                score_trials = []
                sample_dropped = False
                for _ in range(N_episodes):
                    # Run episode, get score, record score
                    score = self.run_episode()
                    score_trials.append(score)
                    if score < kwargs.get("score_drop_threshold", -np.inf):
                        sample_dropped = True
                        break

                # Take mean score of N_episodes, record if best score yet
                mean_score = np.mean(score_trials)
                new_best = (best_score is None) or (mean_score > best_score)
                if new_best and not sample_dropped:
                    best_score = mean_score
                    print(
                        f"New best score {best_score:.3f} in sample {samp_num}"
                    )
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
        except:
            print("\n\nSomething stopped sample loop. Continuing...\n")

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
        self.agent.set_random_weights()

    ################################ saving functions

    def save_params_dict(self):
        """
        For saving the parameters used in a .json file, for later analysis.
        """

        self.run_params["env_name"] = self.env_name
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
        self.dt_str = self.run_params["dt_str"]
        self.run_dir = self.run_params["run_dir"]
        self.agent.NN_type = self.run_params[
            "NN_type"
        ]  # not necessary probably? Be careful

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

    def save_all_sample_stats(self, sample_dict, **kwargs):
        """
        For saving all the stats and plots for the sampling, just a collector
        function.

        """

        self.save_sample_dict(sample_dict)
        if kwargs.get("save_plots", True):
            plot_fns.plot_scores(
                self.run_dir,
                sample_dict,
                self.env_name,
                self.dt_str,
                **self.plot_params,
            )
            plot_fns.plot_all_trial_stats(
                self.run_dir,
                sample_dict,
                self.env_name,
                self.dt_str,
                **self.plot_params,
            )
            plot_fns.plot_sample_histogram(
                self.run_dir,
                sample_dict["all_scores"],
                "Mean sample score",
                f"{self.env_name}_all_scores_dist_{self.dt_str}.png",
                self.env_name,
                self.dt_str,
                plot_log=True,
                **self.plot_params,
                **kwargs,
            )
            plot_fns.plot_weight_stats(
                self.run_dir,
                sample_dict,
                self.env_name,
                self.dt_str,
                **self.plot_params,
            )
            plot_fns.plot_score_percentiles(
                self.run_dir,
                sample_dict,
                self.env_name,
                self.dt_str,
                **self.plot_params,
            )


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

    run_params["run_dir"] = os.path.join(path_utils.get_output_dir(), "tmp")
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
