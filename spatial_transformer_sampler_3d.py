import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function
from chainer.utils import argument
from chainer.utils import type_check

class SpatialTransformerSampler3D(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 == n_in)

        x_type = in_types[0]
        grid_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            grid_type.dtype == x_type.dtype,
            x_type.ndim == 5,
            grid_type.ndim == 5,
            grid_type.shape[1] == 3,
            x_type.shape[0] == grid_type.shape[0],
        )

    def forward_cpu(self, inputs):
        return self._forward(inputs)

    def forward_gpu(self, inputs):
        return self._forward(inputs)

    def _forward(self, inputs):
        x, grid = inputs
        xp = backend.get_array_module(x)
        B, C, H, W, D = x.shape
        _, _, out_H, out_W, out_D = grid.shape

        grid = grid.reshape(grid.shape[:2] + (-1,))

        u = grid[:, 0] # D
        v = grid[:, 1] # W
        w = grid[:, 2] # H

        # Pad the image so that pixels locating outside of the original
        # image's size can be sampled.
        x_pad = xp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), mode='constant')

        # Rescale coordinates from [-1, 1] to [0, width or height - 1],
        # and adjust them to the padded image.
        u = (u + 1) * (D - 1) / 2 + 1
        v = (v + 1) * (W - 1) / 2 + 1
        w = (w + 1) * (H - 1) / 2 + 1

        u_clipped = u.clip(0, D + 1)
        v_clipped = v.clip(0, W + 1)
        w_clipped = w.clip(0, H + 1)

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        u0 = xp.floor(u_clipped).astype(numpy.int32)
        u0 = u0.clip(0, D)
        u1 = u0 + 1
        v0 = xp.floor(v_clipped).astype(numpy.int32)
        v0 = v0.clip(0, W)
        v1 = v0 + 1
        w0 = xp.floor(w_clipped).astype(numpy.int32)
        w0 = w0.clip(0, H)
        w1 = w0 + 1

        # weights
        k1 = (u1 - u_clipped) * (v1 - v_clipped) * (w1 - w_clipped) # 1 1 1
        k2 = (u1 - u_clipped) * (v1 - v_clipped) * (w_clipped - w0) # 1 1 0
        k3 = (u1 - u_clipped) * (v_clipped- v0)  * (w1 - w_clipped) # 1 0 1
        k4 = (u1 - u_clipped) * (v_clipped- v0)  * (w_clipped - w0) # 1 0 0
        k5 = (u_clipped - u0) * (v1 - v_clipped) * (w1 - w_clipped) # 0 1 1
        k6 = (u_clipped - u0) * (v1 - v_clipped) * (w_clipped - w0) # 0 1 0
        k7 = (u_clipped - u0) * (v_clipped- v0)  * (w1 - w_clipped) # 0 0 1
        k8 = (u_clipped - u0) * (v_clipped- v0)  * (w_clipped - w0) # 0 0 0

        k1 = k1.astype(x_pad.dtype, copy=False)
        k2 = k2.astype(x_pad.dtype, copy=False)
        k3 = k3.astype(x_pad.dtype, copy=False)
        k4 = k4.astype(x_pad.dtype, copy=False)
        k5 = k5.astype(x_pad.dtype, copy=False)
        k6 = k6.astype(x_pad.dtype, copy=False)
        k7 = k7.astype(x_pad.dtype, copy=False)
        k8 = k8.astype(x_pad.dtype, copy=False)

        x_indexed_1 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w0[b], v0[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_2 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w1[b], v0[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_3 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w0[b], v1[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_4 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w1[b], v1[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_5 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w0[b], v0[b], u1[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_6 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w1[b], v0[b], u1[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_7 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w0[b], v1[b], u1[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_8 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w1[b], v1[b], u1[b]], axis=0) for b in range(B)], axis=0)

        y =  k1[:, :, None] * x_indexed_1
        y += k2[:, :, None] * x_indexed_2
        y += k3[:, :, None] * x_indexed_3
        y += k4[:, :, None] * x_indexed_4
        y += k5[:, :, None] * x_indexed_5
        y += k6[:, :, None] * x_indexed_6
        y += k7[:, :, None] * x_indexed_7
        y += k8[:, :, None] * x_indexed_8

        y = y.reshape(B, out_H, out_W, out_D, C).transpose(0, 4, 1, 2, 3)
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        return self._backward(inputs, grad_outputs)

    def backward_gpu(self, inputs, grad_outputs):
        return self._backward(inputs, grad_outputs)

    def _backward(self, inputs, grad_outputs):
        x, grid = inputs
        xp = backend.get_array_module(x)
        gy, = grad_outputs

        B, C, H, W, D = x.shape
        _, _, out_H, out_W, out_D = grid.shape

        grid = grid.reshape(grid.shape[:2] + (-1,))

        u = grid[:, 0] # D
        v = grid[:, 1] # W
        w = grid[:, 2] # H

        # Pad the image so that points locating outside of the original
        # image's size can be sampled.
        x_pad = xp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), mode='constant')

        # Rescale coordinates from [-1, 1] to [0, width or height or depth- 1],
        # and adjust them to the padded image.
        u = (u + 1) * (D - 1) / 2 + 1
        v = (v + 1) * (W - 1) / 2 + 1
        w = (w + 1) * (H - 1) / 2 + 1

        u_clipped = u.clip(0, D + 1)
        v_clipped = v.clip(0, W + 1)
        w_clipped = w.clip(0, H + 1)

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        u0 = xp.floor(u_clipped).astype(numpy.int32)
        u0 = u0.clip(0, D)
        u1 = u0 + 1
        v0 = xp.floor(v_clipped).astype(numpy.int32)
        v0 = v0.clip(0, W)
        v1 = v0 + 1
        w0 = xp.floor(w_clipped).astype(numpy.int32)
        w0 = w0.clip(0, H)
        w1 = w0 + 1

        # weights
        ku0 = u_clipped - u0
        ku1 = u1 - u_clipped
        kv0 = v_clipped - v0
        kv1 = v1 - v_clipped
        kw0 = w_clipped - w0
        kw1 = w1 - w_clipped

        ku0 = ku0.astype(gy.dtype, copy=False)
        ku1 = ku1.astype(gy.dtype, copy=False)
        kv0 = kv0.astype(gy.dtype, copy=False)
        kv1 = kv1.astype(gy.dtype, copy=False)
        kw0 = kw0.astype(gy.dtype, copy=False)
        kw1 = kw1.astype(gy.dtype, copy=False)

        # --- gu, gv, gw
        x_indexed_1 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w0[b], v0[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_2 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w1[b], v0[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_3 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w0[b], v1[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_4 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w1[b], v1[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_5 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w0[b], v0[b], u1[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_6 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w1[b], v0[b], u1[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_7 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w0[b], v1[b], u1[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_8 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, w1[b], v1[b], u1[b]], axis=0) for b in range(B)], axis=0)

        gu = -kv1[:, :, None] * kw1[:, :, None] * x_indexed_1
        gu -= kv1[:, :, None] * kw0[:, :, None] * x_indexed_2
        gu -= kv0[:, :, None] * kw1[:, :, None] * x_indexed_3
        gu -= kv0[:, :, None] * kw0[:, :, None] * x_indexed_4
        gu += kv1[:, :, None] * kw1[:, :, None] * x_indexed_5
        gu += kv1[:, :, None] * kw0[:, :, None] * x_indexed_6
        gu += kv0[:, :, None] * kw1[:, :, None] * x_indexed_7
        gu += kv0[:, :, None] * kw0[:, :, None] * x_indexed_8

        gv = -ku1[:, :, None] * kw1[:, :, None] * x_indexed_1
        gv -= ku1[:, :, None] * kw0[:, :, None] * x_indexed_2
        gv += ku1[:, :, None] * kw1[:, :, None] * x_indexed_3
        gv += ku1[:, :, None] * kw0[:, :, None] * x_indexed_4
        gv -= ku0[:, :, None] * kw1[:, :, None] * x_indexed_5
        gv -= ku0[:, :, None] * kw0[:, :, None] * x_indexed_6
        gv += ku0[:, :, None] * kw1[:, :, None] * x_indexed_7
        gv += ku0[:, :, None] * kw0[:, :, None] * x_indexed_8

        gw = -ku1[:, :, None] * kv1[:, :, None] * x_indexed_1
        gw += ku1[:, :, None] * kv1[:, :, None] * x_indexed_2
        gw -= ku1[:, :, None] * kv0[:, :, None] * x_indexed_3
        gw += ku1[:, :, None] * kv0[:, :, None] * x_indexed_4
        gw -= ku0[:, :, None] * kv1[:, :, None] * x_indexed_5
        gw += ku0[:, :, None] * kv1[:, :, None] * x_indexed_6
        gw -= ku0[:, :, None] * kv0[:, :, None] * x_indexed_7
        gw += ku0[:, :, None] * kv0[:, :, None] * x_indexed_8

        gu = gu.reshape(B, out_H, out_W, out_D, C).transpose(0, 4, 1, 2, 3)
        gv = gv.reshape(B, out_H, out_W, out_D, C).transpose(0, 4, 1, 2, 3)
        gw = gw.reshape(B, out_H, out_W, out_D, C).transpose(0, 4, 1, 2, 3)

        gu *= gy
        gv *= gy
        gw *= gy

        gu = xp.sum(gu, axis=1)
        gv = xp.sum(gv, axis=1)
        gw = xp.sum(gw, axis=1)

        # Offsets scaling of the coordinates and clip gradients.
        u_reshaped = u.reshape(gu.shape)
        v_reshaped = v.reshape(gv.shape)
        w_reshaped = w.reshape(gw.shape)

        gu = gu / 2. * (D - 1) * (u_reshaped > 0) * (u_reshaped < (D + 1))
        gv = gv / 2. * (W - 1) * (v_reshaped > 0) * (v_reshaped < (W + 1))
        gw = gw / 2. * (H - 1) * (w_reshaped > 0) * (w_reshaped < (H + 1))

        ggrid = xp.concatenate((gu[:, None], gv[:, None], gw[:, None]), axis=1)

        # --- gx
        if xp is numpy:
            scatter_add = numpy.add.at
        else:
            scatter_add = cuda.cupyx.scatter_add
        gx = xp.zeros_like(x_pad)
        gy = gy.reshape(B, C, -1)

        for b in range(B):

            scatter_add(gx[b], (slice(None), w0[b], v0[b], u0[b]),
                        gy[b] * ku1[b] * kv1[b] * kw1[b])
            scatter_add(gx[b], (slice(None), w0[b], v0[b], u1[b]),
                        gy[b] * ku0[b] * kv1[b] * kw1[b])
            scatter_add(gx[b], (slice(None), w0[b], v1[b], u0[b]),
                        gy[b] * ku1[b] * kv0[b] * kw1[b])
            scatter_add(gx[b], (slice(None), w0[b], v1[b], u1[b]),
                        gy[b] * ku0[b] * kv0[b] * kw1[b])
            scatter_add(gx[b], (slice(None), w1[b], v0[b], u0[b]),
                        gy[b] * ku1[b] * kv1[b] * kw0[b])
            scatter_add(gx[b], (slice(None), w1[b], v0[b], u1[b]),
                        gy[b] * ku0[b] * kv1[b] * kw0[b])
            scatter_add(gx[b], (slice(None), w1[b], v1[b], u0[b]),
                        gy[b] * ku1[b] * kv0[b] * kw0[b])
            scatter_add(gx[b], (slice(None), w1[b], v1[b], u1[b]),
                        gy[b] * ku0[b] * kv0[b] * kw0[b])

        gx = gx[:, :, 1:-1, 1:-1, 1:-1]
        return gx, ggrid


def spatial_transformer_sampler_3d(x, grid, **kwargs):
    """3D Spatial Transformer sampler.
    This is a differentiable volume sampler. With a set of sampling points
    ``grid`` and an input feature map ``x``, this produces a sampled output
    feature map.
    This function currently only supports bilinear interpolation as a sampling
    kernel.
    When coordinates in ``grid`` is outside range :math:`[-1, 1]`, values are
    sampled from a zero padded input volume.
    Notation: here is a notation for dimensionalities.
    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input channels.
    - :math:`h`, :math:`w` and :math:`d` are the height, width and depth of the input volume,
      respectively.
    - :math:`h_O`, :math:`w_O` and :math:`d_O` are the height, width and depth of the output
      volume.
    See detail in the following paper: `Spatial Transformer Networks
    <https://arxiv.org/abs/1506.02025>`_.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable of shape :math:`(n, c_I, h, w, d)`.
        grid (~chainer.Variable): Coordinate variable of shape
            :math:`(n, 3, h_O, w_O, d_O)`. Each coordinate defines the spatial
            location in the input where a sampling kernel is applied to get
            the value at a particular pixel in the output.
            ``grid[idx, :, i, j, k]`` corresponds to the coordinate that is used
            to sample the values for an output pixel at location
            :math:`(i, j, k)`.
            In the second dimension, the first coordinate corresponds to the
            location along the horizontal axis, and the second coordinate
            corresponds to the location along the vertical axis.
            The third coordinate corresponds to the location along the depth axis.
            The coordinate :math:`(-1, -1)` corresponds to the upper-left
            corner of the input volume.
    Returns:
        ~chainer.Variable: Output feature map of shape \
            :math:`(n, c_I, h_O, w_O, d_O)`.
    """
    if kwargs:
        argument.check_unexpected_kwargs(
            kwargs, use_cudnn='The argument "use_cudnn" is not '
            'supported anymore. '
            'Use chainer.using_config(\'use_cudnn\', value) '
            'context where value can be `always`, `never`, or `auto`.')
        argument.assert_kwargs_empty(kwargs)
    return SpatialTransformerSampler3D()(x, grid)
