import path_utils
from Sample import Sample

e = Sample('CartPole-v0', NN='FFNN', N_hidden_layers=0, use_bias=False)
sample_dict = e.sample(500, N_episodes=10, print_samp_num=True)
e.save_all_sample_stats(sample_dict)
