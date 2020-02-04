import path_utils
from Evolve import Evolve

e = Evolve('MountainCar-v0', NN='FFNN_multilayer')

e.agent.NN.print_NN_matrices()

obs = e.env.reset()
print(obs)
action = e.agent.get_action(obs)
print(action)





#
