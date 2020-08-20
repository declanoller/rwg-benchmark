import path_utils
from Sample import Sample

"""
from simplest NN solved by NEAT:

https://www.declanoller.com/wp-content/uploads/2019/01/bestNN_GymAgent_23-01-2019_11-54-48-768x512.png

Seems like this should be included in 1HL, 4HU, *with bias*. Here, that means
having (3+1)*(4) + (4+1)*(1) = 21 weights, pretty hefty.

However, another wrinkle:

If you notice there, the weights are all *very* large! Ranging from
(magnitudes) ~1 to 9. We sample from either a Normal(0, 1) (default) or
Uniform(-1, 1) dist! So it's probably nearly impossible to get numbers that big
from a normal. In addition, even a normal scaled that high would be mostly
centered around 0.

As a quick hack, I think I should try mult'ing the dist by 10, and using a
uniform one.

It works!

"""

e = Sample(
    "Pendulum-v0",
    NN="FFNN",
    N_hidden_layers=1,
    N_hidden_units=4,
    random_dist="uniform",
    random_dist_scaling=10.0,
    use_bias=False,
    max_episode_steps=200,
)
sample_dict = e.sample(10000, N_episodes=15, print_samp_num=True)
e.save_all_sample_stats(sample_dict)

print("\n\nBest score found = {:.2f}\n".format(sample_dict["best_score"]))
