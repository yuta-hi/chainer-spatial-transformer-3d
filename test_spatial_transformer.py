import numpy
import chainer
from chainer import testing
from chainer import gradient_check
from chainer import Variable

from chainer.functions import spatial_transformer_grid \
                            as spatial_transformer_grid_2d
from chainer.functions import spatial_transformer_sampler \
                            as spatial_transformer_sampler_2d

from spatial_transformer_grid_3d import spatial_transformer_grid_3d
from spatial_transformer_sampler_3d import spatial_transformer_sampler_3d


def test_spatial_transformer_grid_2d():

    theta_shape = (4,2,3)
    image_shape = (20,30)

    theta = Variable(numpy.random.randn(*theta_shape).astype(numpy.float64))
    grid = spatial_transformer_grid_2d(theta, image_shape)
    grid.grad = numpy.random.randn(*grid.shape).astype(numpy.float64)
    grid.backward(retain_grad=True)

    def f():
        return spatial_transformer_grid_2d(theta, image_shape).array,

    gx, = gradient_check.numerical_grad(f, (theta.array,), (grid.grad,))
    testing.assert_allclose(gx, theta.grad)


def test_spatial_transformer_sampler_2d():

    grid_shape = (3,2,4,5)
    image_shape = (3,1,4,5)

    image = Variable(numpy.random.randn(*image_shape).astype(numpy.float64))
    grid = Variable(numpy.random.randn(*grid_shape).astype(numpy.float64))

    y = spatial_transformer_sampler_2d(image, grid)
    y.grad = numpy.random.randn(*image_shape).astype(numpy.float64)
    y.backward(retain_grad=True)

    def f():
        return spatial_transformer_sampler_2d(image, grid).array,

    gx, ggrid, = gradient_check.numerical_grad(f, (image.array, grid.array), (y.grad,))
    testing.assert_allclose(gx, image.grad)
    testing.assert_allclose(ggrid, grid.grad)


def test_spatial_transformer_grid_3d():

    theta_shape = (4,3,4)
    volume_shape = (10,20,30)

    theta = Variable(numpy.random.randn(*theta_shape).astype(numpy.float64))
    grid = spatial_transformer_grid_3d(theta, volume_shape)
    grid.grad = numpy.random.randn(*grid.shape).astype(numpy.float64)
    grid.backward(retain_grad=True)

    def f():
        return spatial_transformer_grid_3d(theta, volume_shape).array,

    gx, = gradient_check.numerical_grad(f, (theta.array,), (grid.grad,))
    testing.assert_allclose(gx, theta.grad)


def test_spatial_transformer_sampler_3d():

    grid_shape = (2,3,4,5,2)
    volume_shape = (2,1,4,5,2)

    volume = Variable(numpy.random.randn(*volume_shape).astype(numpy.float64))
    grid = Variable(numpy.random.randn(*grid_shape).astype(numpy.float64))

    y = spatial_transformer_sampler_3d(volume, grid)
    y.grad = numpy.random.randn(*volume_shape).astype(numpy.float64)
    y.backward(retain_grad=True)

    def f():
        return spatial_transformer_sampler_3d(volume, grid).array,

    gx, ggrid, = gradient_check.numerical_grad(f, (volume.array, grid.array), (y.grad,))
    testing.assert_allclose(gx, volume.grad)
    testing.assert_allclose(ggrid, grid.grad)


def test_rotation_2d_forward():

    import cv2

    image = cv2.imread('lenna.png')
    image = image.transpose(2,0,1)[numpy.newaxis]
    image = image.astype(numpy.float32)

    theta = [[0,1,0],[1,0,0]]
    theta = numpy.asarray(theta, numpy.float32)
    theta = theta[numpy.newaxis]


    grid = spatial_transformer_grid_2d(theta, image.shape[2:])
    y = spatial_transformer_sampler_2d(image, grid)
    y = y.transpose(0,2,3,1)

    cv2.imwrite('warped_2d.png', y[0].data)


def test_rotation_3d_forward():

    from chainer_bcnn.data import load_image, save_image
    import glob

    volume_file = glob.glob('*.mhd')[0]
    [volume, spacing] = load_image(volume_file)
    volume = volume.reshape(1,1,*volume.shape)
    volume = volume.astype(numpy.float32)

    theta = [[0,1,0,0],[0,0,1,0],[1,0,0,0]]
    theta = numpy.asarray(theta, numpy.float32)
    theta = theta[numpy.newaxis]

    grid = spatial_transformer_grid_3d(theta, volume.shape[2:])
    y = spatial_transformer_sampler_3d(volume, grid)

    save_image('warped_3d.mhd', y[0,0].data, spacing)


if __name__ == '__main__':

    test_rotation_2d_forward()
    test_rotation_3d_forward()

    test_spatial_transformer_grid_2d()
    test_spatial_transformer_grid_3d()

    test_spatial_transformer_sampler_2d()
    test_spatial_transformer_sampler_3d()


