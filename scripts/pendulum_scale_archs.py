import path_utils
import Statistics


"""

This is for running the same set of runs that are in the paper, except with the
fix for Pendulum-v0 to scale its sampling to a range that might let it solve
it.

It only runs Pendulum, but for each, tries both uniform and normal dists
(uniform should favor larger values), and use_bias=True and False.

"""


N_SAMPLES = 20000
N_EPISODES = 20

COMMON_CONST_ARGS = {
    "NN": "FFNN",
    "random_dist_scaling": 10.0,
    "drop_score_threshold": -500,
}

COMMON_VARY_ARGS = {
    "env_name": ["Pendulum-v0"],
    "random_dist": ["uniform", "normal"],
    "use_bias": [True, False],
}

# ############################## 0 HL
Statistics.run_vary_params(
    {**COMMON_CONST_ARGS, "N_hidden_layers": 0,},
    {**COMMON_VARY_ARGS},
    N_samples=N_SAMPLES,
    N_episodes=N_EPISODES,
)


# ############################## 1 HL
Statistics.run_vary_params(
    {**COMMON_CONST_ARGS, "N_hidden_layers": 1,},
    {**COMMON_VARY_ARGS, "N_hidden_units": [2, 4],},
    N_samples=N_SAMPLES,
    N_episodes=N_EPISODES,
)

# ############################## 2 HL
Statistics.run_vary_params(
    {**COMMON_CONST_ARGS, "N_hidden_layers": 2, "N_hidden_units": 4,},
    {**COMMON_VARY_ARGS},
    N_samples=N_SAMPLES,
    N_episodes=N_EPISODES,
)
