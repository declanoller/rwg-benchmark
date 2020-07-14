from RNN1L import RNN1L
from FFNN_multilayer import FFNN_multilayer
import numpy as np
import itertools


"""
A wrapper class for whichever NN you want to use. Handles issues like the right
output function, scaling, etc, to use for different action spaces.
"""


class Agent:
    def __init__(self, env, **kwargs):

        self.N_inputs = env.reset().size

        self.policy_type = kwargs.get("policy_type", "deterministic")
        assert self.policy_type in [
            "deterministic",
            "stochastic",
        ], "policy_type must be either deterministic or stochastic!"

        # Figure out correct output function (i.e., argmax or identity)
        # and output scaling, given the env type.
        if type(env.action_space).__name__ == "Discrete":
            self.action_space_type = "discrete"
            self.N_actions = env.action_space.n
            self.N_outputs = self.N_actions

            # Deterministic (typical) vs stochastic sample
            if self.policy_type == "stochastic":
                self.output_fn = self.discrete_dist_sample
            else:
                self.output_fn = np.argmax

        elif type(env.action_space).__name__ == "Box":
            self.action_space_type = "continuous"
            # This is so that it can reasonably cover the continuous action
            # space for the environment.
            self.action_scale = env.action_space.high.max()
            self.N_actions = len(env.action_space.sample())

            if self.policy_type == "stochastic":
                self.output_fn = self.continuous_dist_sample
                # Twice the number, to parameterize mean and SD.
                self.N_outputs = 2 * self.N_actions
            else:
                self.output_fn = self.scale_continuous_action
                self.N_outputs = self.N_actions

        # Select the NN class to use.
        NN_types_dict = {"RNN": RNN1L, "FFNN": FFNN_multilayer}

        self.NN_type = kwargs.get("NN", "FFNN")
        assert (
            self.NN_type in NN_types_dict.keys()
        ), "Must supply a valid NN type!"
        self.NN = NN_types_dict[self.NN_type](
            self.N_inputs, self.N_outputs, **kwargs
        )

        self.search_method = kwargs.get("search_method", "RWG")
        self.setup_search_method()

        self.init_episode()

    def setup_search_method(self, **kwargs):

        """
        Here, for grid search, we're going to make each weight take on some set
        of values between -1 and 1. It gets a little tricky, because for
        multilayer NNs, they have to be reshaped carefully.
        """

        self.search_done = False

        if self.search_method in [
            "grid_search",
            "bin_grid_search",
            "sparse_bin_grid_search",
        ]:
            if self.search_method == "grid_search":
                # This will assume that you want to search for weights between
                # values -1 and 1, and you just supply how many values you want
                # it to search in that range. It always makes the value odd to
                # make sure it hits 0.
                self.grid_search_res = kwargs.get("grid_search_res", 5)
                if self.grid_search_res % 2 == 0:
                    self.grid_search_res += 1

            if self.search_method in [
                "bin_grid_search",
                "sparse_bin_grid_search",
            ]:
                # This is basically just a grid search, but for only the values
                # -1, 0, +1.
                self.grid_search_res = 3

            self.N_weights = self.NN.N_weights
            self.N_grid_combos = self.grid_search_res ** self.N_weights
            print(f"{self.N_weights} total weights")

            """
            grid_vals = np.linspace(-1.0, 1.0, self.grid_search_res)
            self.grid_search_idx = 0
            if self.search_method == 'sparse_bin_grid_search': grid_vals =
            np.array(sorted(grid_vals, key=lambda x: np.abs(x)))
            """

            self.N_nonzero = 0
            self.nonzero_gen = itertools.product(
                [-1, 1], repeat=self.N_nonzero
            )
            self.nonzero_tuple = next(self.nonzero_gen)

            self.nonzero_ind_gen = itertools.combinations(
                range(self.N_weights), self.N_nonzero
            )

            print(
                (
                    f"\t==>3^{self.N_weights} = {3**self.N_weights} "
                    "total weight sets to try."
                )
            )
            # Set the weights to the first grid point
            w = self.get_next_weight_set()
            self.set_weights_by_list(w)

    def get_next_weight_set(self):

        try:
            nonzero_ind_tuple = next(self.nonzero_ind_gen)
            arr = np.zeros(self.N_weights)
            for val, ind in zip(self.nonzero_tuple, nonzero_ind_tuple):
                arr[ind] = val
            # counter += 1
            # print(f'\t\t\t=========> {arr}')
            return arr

        except StopIteration:

            try:
                self.nonzero_tuple = next(self.nonzero_gen)
                self.nonzero_ind_gen = itertools.combinations(
                    range(self.N_weights), self.N_nonzero
                )
                # print(f'nonzero_ind_tuple = {nonzero_ind_tuple}')
                return self.get_next_weight_set()
                # counter += 1
                # arr = get_set(nonzero_tuple, nonzero_ind_tuple, N_weights)
                # all_sets.append(arr)
                # print(f'\t\t\t=========> {arr}')

            except StopIteration:
                # print('Done with nonzero_ind_gen!\n')
                if self.N_nonzero < self.N_weights:
                    self.N_nonzero += 1
                    self.nonzero_gen = itertools.product(
                        [-1, 1], repeat=self.N_nonzero
                    )
                    return self.get_next_weight_set()
                else:
                    print("Generators finished!")
                    return None

    def init_episode(self):
        # Initialize the agent for an episode. Should only matter for RNNs.
        self.NN.reset_state()

    def get_action(self, state):

        """
        Takes the output of the NN, gets an action from it.

        """

        NN_output = self.NN.forward(state)
        return self.output_fn(NN_output)

    def scale_continuous_action(self, x):

        """
        For scaling the output of the NN in the case of the deterministic
        policy with a continuous action space. Bounds the output to (-1, 1)
        using tanh, then scales it to (-action bound, action bound).
        """

        return self.action_scale * np.tanh(x)

    def discrete_dist_sample(self, x):

        """
        Given an array of outputs from the NN, runs them through a softmax and
        then samples them from a Categorical distribution (i.e., values 0 to
        N-1), and returns that index.
        """
        softmax_x = np.exp(x) / sum(np.exp(x))
        return np.random.choice(list(range(len(x))), p=softmax_x)

    def continuous_dist_sample(self, x):

        """
        This is for sampling a continuous distribution using the outputs of the
        NN. It's assumed that the NN was set up with twice the number of
        actions: N for the mu's (means) of the distribution, N for the sigma's
        (SD's) of the distribution. It will be set up as [mus, sigmas].

        There's unfortunately a little more choice here in terms of how to
        implement it: first, what distribution? Gaussian is common, but Beta
        might play better with a bounded action range. However, Beta
        distributions can take on strange shapes.

        Second, in the original implementation for continuous action spaces, we
        scale them to the action bounds. It makes sense to do that for here as
        well, but it's not straightforward what to scale.

        Lastly, the mus can typically be real values, but the sigmas must be
        positive. In policy gradient methods they're commonly run through a
        softplus/etc activation to ensure this, but this is also one of many
        choices.

        Here I'm doing the following: for the mus, like the original setup,
        they're run through a tanh and scaled to the action space size. For the
        sigmas, they're simply run through a softplus.

        """

        mus_NN = x[: self.N_actions]
        sigmas_NN = x[self.N_actions :]

        mus = np.tanh(mus_NN) * self.action_scale
        sigmas = np.log(1 + np.exp(sigmas_NN))

        return np.random.normal(loc=mus, scale=sigmas)

    def get_weight_matrix(self):
        # Just a wrapper to return the NN weight matrix.
        return self.NN.weights_matrix

    def get_weights_as_list(self):
        return self.NN.get_weights_as_list()

    def set_weight_matrix(self, w):
        # Used to set the weight matrix, but be careful, because it
        # has to be in the right form for that NN.
        self.NN.set_weights(w)

    def set_weights_by_list(self, w_list):
        # For setting the weights of the NN by just giving it a list, rather
        # than a matrix with the correct number of dims (useful for
        # FFNN_multilayer, for example).
        self.NN.set_weights_by_list(w_list)

    def set_random_weights(self):
        # Wrapper for the NN's function to randomize its weights.
        self.NN.set_random_weights()

    def mutate_grid_search(self):
        # Used to step through the generator used for the grid search.
        w = self.get_next_weight_set()
        # print(w)
        if w is not None:
            self.set_weights_by_list(w)
        else:
            self.search_done = True

    def get_weight_sums(self):
        # Get the L0, L1, L2 weight sums for the NN.
        return self.NN.get_weight_sums()


#
