#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:       https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
             https://github.com/titu1994/keras-efficientnets
             https://arxiv.org/abs/1905.11946
@todo:
@bug:
@brief:      The efficient net model implementation.
------------------------------------------- """


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Dense, Dropout, Activation,\
    Conv2D, Input, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs
import learn.root_model as model
import learn.utils.utils as utils
from enum import Enum
import learn.models.blocks.blockargs as bargs
import learn.models.layers.swish as sw
import learn.models.blocks.mbconvblock as mbb
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid


try:
    import inspect
    _inspect_available = True
except ImportError:
    _inspect_available = False

try:
    from joblib import Parallel, delayed
    _joblib_available = True
except ImportError:
    _joblib_available = False


DEFAULT_DROPOUT_RATE = 0.0
DEFAULT_WIDTH_COEFFICIENT = 0.0
DEFAULT_DEPTH_COEFFICIENT = 0.0
DEFAULT_DEPTH_DIVISOR = 8
DEFAULT_MIN_DEPTH = 0
DEFAULT_BATCH_NORM_MOMENTUM = 0.99
DEFAULT_BATCH_NORM_EPSILON = 1e-3
DEFAULT_DROP_CONNECT_RATE = 0.0


class VersionID(Enum):
    """ Versions of the EfficientNet `BX`, where `BX` refers to the version ID.
    """
    RAW = -1
    DEFAULT = 0
    LATEST = 7
    B0 = 0
    B1 = 1
    B2 = 2
    B3 = 3
    B4 = 4
    B5 = 5
    B6 = 6
    B7 = 7


def get_compound_coeff_func(phi=1.0, max_cost=2.0):
    """ Cost function from the EfficientNets paper to compute candidate values for alpha, beta
    and gamma parameters respectively.

    These values are then used to train models, and the validation accuracy is used to select
    the best base parameter set at phi = 1.

    :param phi: (float) The base power of the parameters. Kept as 1 for initial search of base parameters.
    :param max_cost: (float) The maximum cost of permissible. User defined constant generally set to 2.
    :return A function which accepts a numpy vector of 3 values, and computes the mean squared error between the
            `max_cost` value and the cost computed as `cost = x[0] * (x[1] ** 2) * (x[2] ** 2)`.
    """
    def compound_coeff(x, phi=phi, max_cost=max_cost):
        depth = alpha = x[0]
        width = beta = x[1]
        resolution = gamma = x[2]

        # scale by power. Phi is generally kept as 1.0 during search.
        alpha = alpha ** phi
        beta = beta ** phi
        gamma = gamma ** phi

        # compute the cost function
        cost = alpha * (beta ** 2) * (gamma ** 2)
        return (cost - max_cost) ** 2

    return compound_coeff


def _sequential_optimize(param_grid, param_set, loss_func,
                         num_coeff, ineq_constraints, verbose):
    """
    :param param_grid:
    :param param_set:
    :param loss_func:
    :param num_coeff:
    :param ineq_constraints:
    :param verbose:
    :return:
    """
    param_holder = np.empty((num_coeff,))
    for ix, param in enumerate(param_grid):
        # create a vector for the cost function and minimise using SLSQP
        for i in range(num_coeff):
            param_holder[i] = param[i]
        x0 = param_holder
        res = minimize(loss_func, x0, method='SLSQP', constraints=ineq_constraints)
        param_set[ix] = res.x
        if verbose:
            if (ix + 1) % 1000 == 0:
                print("Computed {:6d} parameter combinations...".format(ix + 1))
    return param_set


def _joblib_optimize(param, loss_func, num_coeff, ineq_constraints):
    """ Minimizes the loss function for given parameters.
    :param param:
    :param loss_func:
    :param num_coeff:
    :param ineq_constraints:
    :return: results
    """
    x0 = np.asarray([param[i] for i in range(num_coeff)])
    res = minimize(loss_func, x0, method='SLSQP', constraints=ineq_constraints)
    return res.x


def optimize_coefficients(num_coeff=3, loss_func=None, phi=1.0, max_cost=2.0,
                          search_per_coeff=4, sort_by_loss=False, save_coeff=True,
                          tol=None, verbose=True):
    """ Computes the possible values of any number of coefficients, given a cost function, phi and max cost permissible.

    Takes into account the search space per coefficient so that the subsequent grid search does not become
    prohibitively large.

    :param num_coeff: number of coefficients that must be optimized.
    :param cost_func: coefficient cost function that minimised to satisfy the least squares solution. The function can
            be user defined, in which case it must accept a numpy vector of length `num_coeff` defined above.
            It is suggested to use MSE against a pre-refined `max_cost`.
    :param phi: The base power of the parameters. Kept as 1 for initial search of base parameters.
    :param max_cost: The maximum cost of permissible. User defined constant generally set to 2.
    :param search_per_coeff: int declaring the number of values tried per coefficient. Constructs a search space
            of size `search_per_coeff` ^ `num_coeff`.
    :param sort_by_loss: bool. Whether to sort the result set by its loss value, in order of lowest loss first.
    :param save_coeff: bool, whether to save the resulting coefficients into the file
           `param_coeff.npy` in current working dir.
    :param tol: float tolerance of error in the cost function. Used to select candidates which have a cost
             less than the tolerance.
    :param verbose: bool, whether to print messages during execution.
    :return A numpy array of shape [search_per_coeff ^ num_coeff, num_coeff], each row defining the value of the
            coefficients which minimise the cost function satisfactorily (to some machine precision).
    """
    phi = float(phi)
    max_cost = float(max_cost)
    search_per_coeff = int(search_per_coeff)

    # if user defined cost function is not provided, use the one from the paper in reference.
    if loss_func is None:
        loss_func = get_compound_coeff_func(phi, max_cost)

    # prepare inequality constraints
    ineq_constraints = {
        'type': 'ineq',
        'fun': lambda x: x - 1.
    }

    # Prepare a matrix to store results
    num_samples = search_per_coeff ** num_coeff
    param_range = [num_samples, num_coeff]

    # sorted by ParameterGrid acc to its key value, assuring sorted behaviour for Python < 3.7.
    grid = {i: np.linspace(1.0, max_cost, num=search_per_coeff)
            for i in range(num_coeff)}

    if verbose:
        print("Preparing parameter grid...")
        print("Number of parameter combinations :", num_samples)

    param_grid = ParameterGrid(grid)

    if _joblib_available:
        with Parallel(n_jobs=-1, verbose=10 if verbose else 0) as parallel:
            param_set = parallel(delayed(_joblib_optimize)(param, loss_func, num_coeff, ineq_constraints)
                                 for param in param_grid)

        param_set = np.asarray(param_set)
    else:
        if verbose and num_samples > 1000:
            print("Consider using `joblib` library to speed up sequential "
                  "computation of {} combinations of parameters".format(num_samples))

        param_set = np.zeros(param_range)
        param_set = _sequential_optimize(param_grid, param_set, loss_func,
                                         num_coeff=num_coeff,
                                         ineq_constraints=ineq_constraints,
                                         verbose=verbose)

    # compute a minimum tolerance of the cost function to select it in the candidate list.
    if tol is not None:
        if verbose:
            print("Filtering out samples below tolerance threshold...")

        tol = float(tol)
        cost_scores = np.asarray([loss_func(xi) for xi in param_set])
        param_set = param_set[np.where(cost_scores <= tol)]
    else:
        cost_scores = None

    # sort by lowest loss first
    if sort_by_loss:
        if verbose:
            print("Sorting by loss...")

        if cost_scores is None:
            cost_scores = ([loss_func(xi) for xi in param_set])
        else:
            cost_scores = cost_scores.tolist()

        cost_scores_id = [(idx, loss) for idx, loss in enumerate(cost_scores)]
        cost_scores_id = sorted(cost_scores_id, key=lambda x: x[1])

        ids = np.asarray([idx for idx, loss in cost_scores_id])
        # reorder the original param set
        param_set = param_set[ids, ...]

    if save_coeff:
        np.save('param_coeff.npy', param_set)

    return param_set


def optimize_effnet():
    """ Optimizes the Efficient net.
    """
    def cost_func_wrapper(phi=1.0, max_cost=2.0):
        def cost_func(x: np.ndarray, phi=phi, max_cost=max_cost) -> float:
            depth = x[0] ** phi
            width = x[1] ** phi
            kernel_width = x[2] ** phi

            cost = (depth * width ** 2 * kernel_width ** 0.5)
            loss = (cost - max_cost) ** 2
            return loss
        return cost_func

    phi = 1.0
    loss_func = cost_func_wrapper(phi=phi, max_cost=2.0)

    results = optimize_coefficients(num_coeff=3, loss_func=loss_func,
                                    phi=1.0, max_cost=2.0, search_per_coeff=25,
                                    save_coeff=False, tol=None, sort_by_loss=True)

    print("Num unique configs = ", len(results))
    for i in range(10):  # print just the first 10 results out of 1000 results
        print(i + 1, results[i], "Cost :", loss_func(results[i]))

    phi = 4.0
    # params = [1.84, 1.007, 1.15]
    params = [1.04163396, 1.33328223, 1.16665207]
    cost = np.sqrt(loss_func(params, phi=phi, max_cost=0.))
    print("x0", params[0] ** phi)
    print("x1", params[1] ** phi)
    print("x2", params[2] ** phi)
    print(cost)


def efficientnet_factory(
        input_shape: list,
        output_shape: list,
        id=VersionID.LATEST,
        weight_links=None,
        optimizer=utils.Optimizer.ADADELTA.value,
        loss=utils.Loss.SPARSE_CAT_CROSS_ENTROPY.value,
        metrics=utils.Metrics.SPARSE_CAT_ACCURACY.value,
        log_path=None,
        ckpt_path=None,
        include_top=True,
        layer_prefix=""):
    """ Init. method.
    :param include_top: (bool) Include top layers.
    :param layer_prefix: (str) Add this prefix to ALL layer names.
    :param weight_links: (dict) This dictionary contains the link to the weights.
    :param input_shape: (list) Input shape of the input data (W x H x D).
    :param output_shape: (list) The output shape for the output data.
    :param optimizer: (utils.Optimizer) The optimizer.
    :param loss: (utils.Loss) The loss.
    :param metrics: (utils.Metrics) The evaluation metric or metrics.
    :param log_path: (str) The path to the desired log directory.
    :param ckpt_path: (str) The path to the checkpoint directory.
    :param id: (VersionID) The version ID of the model.
    :return model (EfficientNet): The configured model.
    :raises ValueError if the passed id is not part of VersionID.
            ValueError if the version specific image size does not fit the passed input shape.
    """
    if type(id) == VersionID:
        id = id.value
    if id == VersionID.B0.value:
        width_coefficient = 1.0
        depth_coefficient = 1.0
        default_size = 224
        dropout_rate = 0.2
    elif id == VersionID.B1.value:
        width_coefficient = 1.0
        depth_coefficient = 1.1
        default_size = 240
        dropout_rate = 0.2
    elif id == VersionID.B2.value:
        width_coefficient = 1.1
        depth_coefficient = 1.2
        default_size = 260
        dropout_rate = 0.3
    elif id == VersionID.B3.value:
        width_coefficient = 1.2
        depth_coefficient = 1.4
        default_size = 300
        dropout_rate = 0.3
    elif id == VersionID.B4.value:
        width_coefficient = 1.4
        depth_coefficient = 1.8
        default_size = 380
        dropout_rate = 0.4
    elif id == VersionID.B5.value:
        width_coefficient = 1.6
        depth_coefficient = 2.2
        default_size = 456
        dropout_rate = 0.4
    elif id == VersionID.B6.value:
        width_coefficient = 1.8
        depth_coefficient = 2.6
        default_size = 528
        dropout_rate = 0.5
    elif id == VersionID.B7.value:
        width_coefficient = 2.0
        depth_coefficient = 3.1
        default_size = 600
        dropout_rate = 0.5
    else:
        raise ValueError("{0} is not defined within VersionID!".format(id))
    if default_size not in input_shape:
        raise ValueError("The image size defined in the `input_shape`"
                         " has to be {0} and not the size contained in {1}!"
                         .format(default_size, input_shape))
    return EfficientNet(
        include_top=include_top,
        dropout_rate=dropout_rate,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        input_shape=input_shape,
        output_shape=output_shape,
        weight_links=weight_links,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        log_path=log_path,
        ckpt_path=ckpt_path,
        layer_prefix=layer_prefix,
        block_args=bargs.get_default_blockargs())


class EfficientNetConvInitializer(tf.keras.initializers.Initializer):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas base_path we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
    """
    def __init__(self):
        """ Init. method.
        """
        super(EfficientNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        """ Call method.
        :param shape:
        :param dtype:
        :return: an initialization for the variable
        """
        dtype = dtype or tf.keras.backend.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.keras.backend.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


class EfficientNetDenseInitializer(tf.keras.initializers.Initializer):
    """Initialization for dense kernels.
        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out', distribution='uniform').
        It is written out explicitly base_path for clarity.
    """
    def __init__(self):
        """ Init. method.
        """
        super(EfficientNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        """ Call method.
        :param shape:
        :param dtype:
        :return: an initialization for the variable
        """
        dtype = dtype or tf.keras.backend.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return tf.keras.backend.random_uniform(shape, -init_range, init_range, dtype=dtype)


class EfficientNet(model.Model):
    """ Efficient Net.
    """

    def __init__(self,
                 input_shape: list,
                 output_shape: list,
                 weight_links=None,
                 optimizer=utils.Optimizer.ADADELTA.value,
                 loss=utils.Loss.SPARSE_CAT_CROSS_ENTROPY.value,
                 metrics=utils.Metrics.SPARSE_CAT_ACCURACY.value,
                 log_path=None,
                 ckpt_path=None,
                 layer_prefix="",
                 data_format=None,
                 block_args=None,
                 width_coefficient=DEFAULT_WIDTH_COEFFICIENT,
                 depth_coefficient=DEFAULT_DEPTH_COEFFICIENT,
                 batch_norm_momentum=DEFAULT_BATCH_NORM_MOMENTUM,
                 batch_norm_epsilon=DEFAULT_BATCH_NORM_EPSILON,
                 drop_connect_rate=DEFAULT_DROP_CONNECT_RATE,
                 dropout_rate=DEFAULT_DROPOUT_RATE,
                 include_top=True,
                 **kwargs):
        """ Init. method.
        :param data_format (list): The format of the image shape.
        :param layer_prefix (str): Add this prefix to ALL layer names.
        :param weight_links (dict): This dictionary contains the link to the weights.
        :param input_shape (list): Input shape of the input data (W x H x D).
        :param output_shape (list): The output shape for the output data.
        :param optimizer (utils.Optimizer): The optimizer.
        :param loss (utils.Loss): The loss.
        :param metrics (utils.Metrics): The evaluation metric or metrics.
        :param log_path (str): The path to the desired log directory.
        :param ckpt_path (str): The path to the checkpoint directory.
        :param block_args (list): Block arguments.
        :param width_coefficient (float):
        :param depth_coefficient (float):
        :param batch_norm_momentum (float):
        :param batch_norm_epsilon (float):
        :param drop_connect_rate (float):
        :param dropout_rate (float):
        :param include_top (bool): Include top layers.
        """
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            ckpt_path=ckpt_path,
            log_path=log_path)

        self._batch_norm_momentum = DEFAULT_BATCH_NORM_MOMENTUM
        self._batch_norm_epsilon = DEFAULT_BATCH_NORM_EPSILON
        self._drop_connect_rate = DEFAULT_DROP_CONNECT_RATE
        self._block_args = []
        self._include_top = include_top
        self._dropout_rate = dropout_rate
        self._block_args = block_args if block_args is not None else bargs.get_default_blockargs()
        self._width_coefficient = width_coefficient
        self._depth_coefficient = depth_coefficient
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._drop_connect_rate = drop_connect_rate
        self._data_format = data_format if data_format is not None else tf.keras.backend.image_data_format()
        self._channel_axis = 1 if self._data_format == 'channels_first' else -1
        self._layer_prefix = layer_prefix
        self._optimizer = optimizer
        self._metrics = metrics
        self._loss = loss
        self._weight_links = weight_links
        self._depth_divisor = DEFAULT_DEPTH_DIVISOR
        self._min_depth = DEFAULT_MIN_DEPTH
        self._model = self._build_model()
        self._model = self._setup_weights()
        self._configure(optimizer=optimizer, loss=loss, metrics=metrics)

    def _construct_model(self) -> (tf.Tensor, tf.Tensor):
        """ Adds layers to the learn.
        :return input and output tensor
        """
        input_tensor = None
        if input_tensor is None:
            inputs = Input(shape=self._input_shape)
        else:
            if not tf.is_tensor(input_tensor):
                inputs = Input(tensor=input_tensor, shape=self._input_shape)
            else:
                inputs = input_tensor

        x = inputs
        x = Conv2D(
                filters=utils.round_filters(32, self._width_coefficient, self._depth_divisor, self._min_depth),
                kernel_size=[3, 3], strides=[2, 2], kernel_initializer=EfficientNetConvInitializer(),
                padding='same', use_bias=False)(x)
        x = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)(x)
        x = sw.Swish()(x)

        if self._block_args is not None and len(self._block_args) > 0:
            num_blocks = sum([block_args.num_repeat for block_args in self._block_args])
            drop_connect_rate_per_block = self._drop_connect_rate / float(num_blocks)

            # Blocks part
            for block_idx, block_args in enumerate(self._block_args):
                assert block_args.num_repeat > 0

                # Update block input and output filters based on depth multiplier.
                block_args.input_filters = utils.round_filters(
                    block_args.input_filters, self._width_coefficient, self._depth_divisor, self._min_depth)
                block_args.output_filters = utils.round_filters(
                    block_args.output_filters, self._width_coefficient, self._depth_divisor, self._min_depth)
                block_args.num_repeat = utils.round_repeats(
                    block_args.num_repeat, self._depth_coefficient)

                # The first block needs to take care of stride and filter size increase.
                x = mbb.MBConvBlock(
                    block_args.input_filters, block_args.output_filters,
                    block_args.kernel_size, block_args.strides,
                    block_args.expand_ratio, block_args.se_ratio,
                    block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                    self._batch_norm_momentum, self._batch_norm_epsilon, self._data_format)(x)

                if block_args.num_repeat > 1:
                    block_args.input_filters = block_args.output_filters
                    block_args.strides = [1, 1]

                for _ in range(block_args.num_repeat - 1):
                    x = mbb.MBConvBlock(
                        block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        self._batch_norm_momentum, self._batch_norm_epsilon, self._data_format)(x)

        # Head part
        x = Conv2D(
            filters=utils.round_filters(1280, self._width_coefficient, self._depth_divisor, self._min_depth),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)(x)
        x = sw.Swish()(x)

        if self._include_top:
            x = GlobalAveragePooling2D(data_format=self._data_format)(x)
            if self._dropout_rate > 0:
                x = Dropout(self._dropout_rate)(x)
            x = Dense(self._output_shape[0], kernel_initializer=EfficientNetDenseInitializer())(x)
            x = Activation('softmax')(x)
        else:
            #x = GlobalAveragePooling2D()(x)
            x = GlobalMaxPooling2D()(x)
        outputs = x

        # Ensure that the models takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)

        return inputs, outputs

    def _build_model(self) -> Model:
        """ Builds the model
        :return model: The composed model.
        """
        model_in, model_out = self._construct_model()
        self._model = Model(inputs=model_in, outputs=model_out)
        return self._model

    def _setup_weights(self) -> Model:
        """ Setup for weights.
        :return: model
        """
        if self._weight_links is None:
            print("No pre-trained weights loaded ...")
        else:
            print("Loading pre-trained weights ...")
            self._load_weights(self._weight_links)
        return self._model
