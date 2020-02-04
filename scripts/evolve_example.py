import path_utils
from Evolve import Evolve

#e = Evolve('Acrobot-v1', NN='FFNN', N_hidden_layers=1, N_hidden_units=4)
e = Evolve('Acrobot-v1', NN='FFNN', N_hidden_layers=0, use_bias=False)
#e = Evolve('MountainCar-v0', NN='FFNN', N_hidden_layers=1, N_hidden_units=2, search_method='sparse_bin_grid_search')
#evo_dict = e.evolve(100, N_trials=5, print_gen=True, save_all_weights=True)
evo_dict = e.evolve(20, N_trials=2, print_gen=True)
e.record_best_episode(evo_dict['best_weights'])

exit()
e.save_all_evo_stats(evo_dict)
#e = Evolve('MountainCar-v0', NN='FFNN_multilayer', N_hidden_layers=1)
#e = Evolve('Pendulum-v0', NN='FFNN_multilayer', N_hidden_layers=1, search_method='sparse_bin_grid_search')
#e = Evolve('Pendulum-v0', NN='FFNN_multilayer', N_hidden_layers=1, N_hidden_units=2, search_method='RWG')
#e = Evolve('MountainCar-v0', NN='FFNN_multilayer', N_hidden_layers=0, search_method='RWG')
#e = Evolve('MountainCar-v0', NN='FFNN')
