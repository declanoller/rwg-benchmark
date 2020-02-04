import path_utils
import numpy as np
from RNN1L import RNN1L
from FFNN_multilayer import FFNN_multilayer
import gym
import matplotlib.pyplot as plt
import os

# Environment setup
#env_name = "CartPole-v0"
env_name = 'Acrobot-v1'
env = gym.make(env_name)
nactions = env.action_space.n
ninputs = env.reset().size

# Network setup
#net = RNN1L(ninputs, nactions)
net = FFNN_multilayer(ninputs, nactions) # Simple feedforward NN
net.set_random_weights()

def run_episode(ep_net):
    # Function for running a single episode, returns score.
    ep_net.reset_state()
    obs = env.reset()
    score = 0
    done = False
    step = 0
    while not done:
        #print(f'step {step}')
        action = ep_net.get_action(obs)
        obs, rew, done, info = env.step(action)
        score += rew
        step += 1
    return score


all_scores = []
best_scores = []
best_score = None
best_weights = None
max_gen = 1000
N_trials = 3

# Gameplay loop
for gen in range(max_gen):

    if gen % max(1, max_gen//100) == 0:
        print(f'generation {gen}')



    # Get new NN
    net.set_random_weights()

    # Run each agent for several trials, to get a representative mean
    score_trials = []
    for _ in range(N_trials):
        ep_score = run_episode(net)
        score_trials.append(ep_score)

    # Take mean, append, test to see if best yet.
    mean_score = np.mean(score_trials)
    all_scores.append(mean_score)
    # I use >= here, because it's possible that the N_trials could achieve
    # the maximum score but still fail the 100 episode average, so I want it
    # to be able to try again with another NN.
    if len(best_scores)==0 or mean_score >= best_score:
        best_score = mean_score
        best_weights = net.weights_matrix
        # If it achieved a new best score, test for 100 episode average score.
        # If 100 ep mean score is >= 195.0, it's considered solved.
        eval_trials = []
        for _ in range(100):
            score = run_episode(net)
            eval_trials.append(score)
        eval_mean = np.mean(eval_trials)
        if eval_mean >= 195.0:
            print(f'Solved! 100 episode mean score = {eval_mean:.2f} in generation {gen}')
            break
        else:
            print(f'Unsolved. 100 episode mean score = {eval_mean:.2f} in generation {gen}')

    # Append even if there was no improvement
    best_scores.append(best_score)


# Plot results
print(f'Best score achieved: {best_score}')
print(f'Best weight matrix: \n{best_weights}')

plt.plot(best_scores, color='tomato', label='Best FF found')
plt.plot(all_scores, color='dodgerblue', label='All FF')

plt.xlabel('Episode')
plt.ylabel('Fitness Function (FF)')
plt.legend()
plt.title('CartPole-v0 environment')
plt.savefig(os.path.join(path_utils.get_output_dir(), 'NE_cartpole_FF.png'))
plt.show()

#env = gym.wrappers.Monitor(env, 'video', force = True)
# Set to best weights found, run episode and show
net.set_weights(best_weights)
obs = env.reset()
score = 0
done = False
while not done:
    env.render()
    action = net.get_action(obs)
    obs, rew, done, info = env.step(action)
    score += rew

print(f'Final score: {score}')
env.close()


#
