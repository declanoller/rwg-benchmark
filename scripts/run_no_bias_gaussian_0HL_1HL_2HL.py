import path_utils
import Statistics


"""
For reproducing the figures from the paper.
"""


N_SAMPLES = 10
N_EPISODES = 2


# ############################## 0 HL
Statistics.run_vary_params(
    {"NN": "FFNN", "N_hidden_layers": 0, "use_bias": False},
    {
        "env_name": [
            "CartPole-v0",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
            "Pendulum-v0",
            "Acrobot-v1",
        ]
    },
    N_samples=N_SAMPLES,
    N_episodes=N_EPISODES,
)


# ############################## 1 HL
Statistics.run_vary_params(
    {"NN": "FFNN", "N_hidden_layers": 1, "use_bias": False},
    {
        "env_name": [
            "CartPole-v0",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
            "Pendulum-v0",
            "Acrobot-v1",
        ],
        "N_hidden_units": [2, 4],
    },
    N_samples=N_SAMPLES,
    N_episodes=N_EPISODES,
)

# ############################## 2 HL
Statistics.run_vary_params(
    {
        "NN": "FFNN",
        "N_hidden_layers": 2,
        "N_hidden_units": 4,
        "use_bias": False,
    },
    {
        "env_name": [
            "CartPole-v0",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
            "Pendulum-v0",
            "Acrobot-v1",
        ],
    },
    N_samples=N_SAMPLES,
    N_episodes=N_EPISODES,
)
