import path_utils
from Sample import Sample

"""

This script tests using a stochastic policy output from the NN, rather than a
deterministic one.

For ex: typically in RWG, for a discrete action space, the action selected is
just the argmax of the output units (like in Q-learning). However, you could
alternatively do it the way a common discrete space policy gradient algo would:
by running the NN outputs through a softmax and sampling the action with
respect to those probabilities.

Will this have different behavior than the original case? Let's see!

In the case of continuous action space, we can try something like
parameterizing a Gaussian action distribution and sampling from that.

"""


e = Sample(
    "CartPole-v0",
    NN="FFNN",
    N_hidden_layers=1,
    N_hidden_units=4,
    policy_type="stochastic",
)
sample_dict = e.sample(1000, N_episodes=20, print_samp_num=True)
e.save_all_sample_stats(sample_dict)

print("\n\nBest score found = {:.2f}\n".format(sample_dict["best_score"]))
