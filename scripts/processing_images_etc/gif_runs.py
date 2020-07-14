import path_utils
from Sample import Sample
import os

# env_name = 'CartPole-v0'
# env_name = 'MountainCarContinuous-v0'
# env_name = 'Acrobot-v1'
env_name = "Pendulum-v0"

base_dir = os.path.join(
    path_utils.get_output_dir(),
    "make_gifs_" + env_name + "_" + path_utils.get_date_str(),
)
os.mkdir(base_dir)

e = Sample(
    env_name,
    NN="FFNN",
    N_hidden_layers=1,
    N_hidden_units=4,
    use_bias=True,
    base_dir=base_dir,
    random_dist="uniform",
    random_dist_scaling=10.0,
)

sample_dict = e.sample(10000, N_episodes=10, print_samp_num=True)
e.save_all_sample_stats(sample_dict)


print(sample_dict["best_weights"])

e.record_best_episode(sample_dict["best_weights"])
