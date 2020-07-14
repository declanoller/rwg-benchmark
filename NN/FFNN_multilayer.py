import numpy as np


class FFNN_multilayer:
    def __init__(self, N_inputs, N_outputs, **kwargs):

        act_fn = kwargs.get("act_fn", "tanh")
        random_dist = kwargs.get("random_dist", "normal")
        use_bias = kwargs.get("use_bias", True)
        random_dist_scaling = kwargs.get("random_dist_scaling", 1.0)

        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        self.N_hidden_layers = kwargs.get("N_hidden_layers", 1)
        self.N_hidden_units = kwargs.get("N_hidden_units", N_inputs)

        # MUST DO THIS BEFORE INIT_WEIGHTS!
        random_dists = ["normal", "uniform"]
        assert (
            random_dist in random_dists
        ), "Must supply valid random dist name!"
        self.random_dist = random_dist
        self.random_dist_scaling = random_dist_scaling

        # MUST DO THIS BEFORE INIT_WEIGHTS!
        use_bias_options = [True, False]
        assert (
            use_bias in use_bias_options
        ), "Must supply True or False to use_bias!"
        self.use_bias = use_bias

        self.init_weights()

        activation_fn_d = {
            "tanh": np.tanh,
            "linear": lambda x: x,
            "relu": lambda x: np.maximum(0, x),
        }
        assert (
            act_fn in activation_fn_d.keys()
        ), "Must supply valid activation function name!"
        self.act_fn = activation_fn_d[act_fn]

    def reset_state(self):
        # Nothing to do for FFNN, but want it to be callable still.
        pass

    def print_NN_matrices(self):

        """
        For debugging. Print NN weights/shapes.
        """

        print("N_inputs: ", self.N_inputs)

        for i, w in enumerate(self.weights_matrix):
            print(f"W_{i} shape: {w.shape}")

        print("N_outputs: ", self.N_outputs)

        print("\nWeight matrices:")
        for i, w in enumerate(self.weights_matrix):
            print(f"\nW_{i}:")
            print(w)

    def init_weights(self):

        self.weights_matrix = []

        mat_input_size = self.N_inputs
        if self.use_bias:
            mat_input_size += 1

        """
        Here, we're using the convention of doing W*x, if W is the weight
        matrix and x is vector. So W has dimensions [N_outputs x N_inputs] and
        x has dimensions [N_inputs x 1]. So W*x has dimensions [N_outputs x 1].

        So it composes right to left, like W_2*tanh(W_1*x).

        Note also that we insert a bias unit in each layer.
        """
        for i in range(self.N_hidden_layers):

            mat_output_size = self.N_hidden_units
            if self.random_dist == "normal":
                mat = np.random.randn(mat_output_size, mat_input_size)
            elif self.random_dist == "uniform":
                mat = np.random.uniform(
                    -1.0, 1.0, (mat_output_size, mat_input_size)
                )
            else:
                raise
            self.weights_matrix.append(self.random_dist_scaling * mat)
            mat_input_size = mat_output_size
            if self.use_bias:
                mat_input_size += 1

        # And for the last layer:
        if self.random_dist == "normal":
            mat = np.random.randn(self.N_outputs, mat_input_size)
        elif self.random_dist == "uniform":
            mat = np.random.uniform(
                -1.0, 1.0, (self.N_outputs, mat_input_size)
            )
        else:
            raise

        self.weights_matrix.append(self.random_dist_scaling * mat)

        self.w_mat_shapes = [w.shape for w in self.weights_matrix]
        self.w_mat_lens = [len(w.flatten()) for w in self.weights_matrix]
        self.N_weights = sum(self.w_mat_lens)

    def set_weights(self, weights):
        # Directly set the weights matrix. In the case of the FFNN, it will
        # be a list of np matrices.
        self.weights_matrix = weights

    def set_weights_by_list(self, w_list):
        # Handy for setting externally, when we don't want to deal with the
        # shapes of the matrices of different layers.
        cur_list_idx = 0
        w_mat_list = []
        for w_len, w_shape in zip(self.w_mat_lens, self.w_mat_shapes):
            w_mat = np.array(
                w_list[cur_list_idx : cur_list_idx + w_len]
            ).reshape(w_shape)
            w_mat_list.append(w_mat)
            cur_list_idx += w_len

        self.set_weights(w_mat_list)

    def get_weights_as_list(self):
        w_mat_list = []
        for w_mat in self.weights_matrix:
            w_flat = w_mat.flatten().tolist()
            w_mat_list += w_flat

        return w_mat_list

    def set_random_weights(self):
        # Just calls init_weights, which will randomize them.
        self.init_weights()

    def forward(self, input_vec):

        """
        Function for giving input to NN, getting output. Matrix multiplies the
        weight matrix at each layer by the input from the last layer (or
        initial input), plus a bias term of 1.0, then applies the nonlinear
        activation function.

        What is done with the output (argmax, etc) is up to the Agent class to
        do.
        """

        x = input_vec

        for i, w in enumerate(self.weights_matrix):
            if self.use_bias:
                x = np.concatenate((x, [1.0]))
            x = np.dot(w, x)
            x = self.act_fn(x)

        return x

    def get_weight_sums(self):

        L0 = sum([np.sum(w) for w in self.weights_matrix]) / self.N_weights
        L1 = (
            sum([np.abs(w).sum() for w in self.weights_matrix])
            / self.N_weights
        )
        L2 = (
            sum([(w ** 2).sum() for w in self.weights_matrix]) / self.N_weights
        )

        return {"L0": L0, "L1": L1, "L2": L2}


#
