import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function
from chainer.utils import argument
from chainer.utils import type_check

class SpatialTransformerGrid3D(function.Function):

    def __init__(self, output_shape):
        self.output_shape = output_shape

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('theta',))

        theta_type = in_types[0]
        type_check.expect(
            theta_type.dtype.kind == 'f',
            theta_type.ndim == 3,
            theta_type.shape[1] == 3,
            theta_type.shape[2] == 4,
        )

    def _forward(self, inputs):
        theta, = inputs
        H, W, D = self.output_shape
        B, _, _ = theta.shape
        xp = backend.get_array_module(theta)

        zs, ys, xs = xp.meshgrid(
            xp.linspace(-1, 1, H, dtype=theta.dtype),
            xp.linspace(-1, 1, W, dtype=theta.dtype),
            xp.linspace(-1, 1, D, dtype=theta.dtype), indexing='ij',
            copy=False
        )

        coords = xp.concatenate(
            [xs[None], ys[None], zs[None], xp.ones((1, H, W, D), dtype=theta.dtype)],
            axis=0)
        grid = theta.dot(coords.reshape(4, H * W * D)).reshape(B, 3, H, W, D)
        return grid,

    def _backward(self, inputs, grad_outputs):
        theta, = inputs
        ggrid, = grad_outputs
        H, W, D = self.output_shape
        B, _, _ = theta.shape
        xp = backend.get_array_module(theta)

        zs, ys, xs = xp.meshgrid(
            xp.linspace(-1, 1, H, dtype=theta.dtype),
            xp.linspace(-1, 1, W, dtype=theta.dtype),
            xp.linspace(-1, 1, D, dtype=theta.dtype), indexing='ij',
            copy=False
        )

        coords = xp.concatenate(
            [xs[None], ys[None], zs[None], xp.ones((1, H, W, D), dtype=theta.dtype)],
            axis=0)
        coords_T = coords.reshape(4, H * W * D).transpose(1, 0)
        ggrid = ggrid.reshape(B, 3, H * W * D)
        gtheta = ggrid.dot(coords_T).reshape(B, 3, 4)
        return gtheta,

    def forward_cpu(self, inputs):
        return self._forward(inputs)

    def backward_cpu(self, inputs, grad_outputs):
        return self._backward(inputs, grad_outputs)

    def forward_gpu(self, inputs):
        return self._forward(inputs)

    def backward_gpu(self, inputs, grad_outputs):
        return self._backward(inputs, grad_outputs)

def spatial_transformer_grid_3d(theta, output_shape, **kwargs):
    """3D Spatial Transformer grid.
    This function generates coordinates of the points sampled from an volume
    to perform warping described in `Spatial Transformer Networks
    <https://arxiv.org/abs/1506.02025>`_.
    Given a coordinate in the warped volume :math:`(x_i^t, y_i^t, z_i^t)`, the point
    sampled from the source volume :math:`(x_i^s, y_i^s, z_i^s)` are calculated
    by the following equation.
    .. math::
        \\left(\\begin{matrix} x_i^s \\\\
            y_i^s \\\\ z_i^s \\end{matrix}\\right)
        =
        \\left(\\begin{matrix} \\theta_{11} & \\theta_{12} & \\theta_{13} & \\theta_{14} \\\\
            \\theta_{21} & \\theta_{22} & \\theta_{23} & \\theta_{24} \\\\
            \\theta_{31} & \\theta_{32} & \\theta_{33} & \\theta_{34} \\end{matrix}\\right)
        \\left(\\begin{matrix} x_i^t \\\\
            y_i^t \\\\ z_i^t \\\\ 1 \\end{matrix}\\right)
    Notation: here is a notation for dimensionalities.
    - :math:`n` is the batch size.
    - :math:`h_O`, :math:`w_O` and :math:`d_O` are the height, width and depth of the output
      volume.
    Args:
        theta (:class:`~chainer.Variable` or :ref:`ndarray`):
            An array of shape :math:`(n, 3, 4)`.
            This is a batch of :math:`3 \\times 4` matrix used for
            the warping described above.
        output_shape (tuple): A tuple of 3 elements: :math:`h_O, w_O, d_O`.
    Returns:
        ~chainer.Variable:  A variable of shape :math:`(n, 3, h_O, w_O, d_O)`.
        In the 2nd dimension, the first element is the coordinate along the
        x axis, and the second element is the coordinate along the y axis.
        The third element is the coordinate along the z axis.
        All the coordinates in the volume are scaled to fit range
        :math:`[-1, 1]`.
        This means that the coordinate :math:`(-1, -1)` corresponds to
        the upper-left corner of the input volume.
    """
    if kwargs:
        argument.check_unexpected_kwargs(
            kwargs, use_cudnn='The argument "use_cudnn" is not '
            'supported anymore. '
            'Use chainer.using_config(\'use_cudnn\', value) '
            'context where value can be `always`, `never`, or `auto`.')
        argument.assert_kwargs_empty(kwargs)
    return SpatialTransformerGrid3D(output_shape)(theta)


