import cv2
import numpy as np

from opencv_convolution_applier import OpenCVConvolutionApplier
from simple_convolution_applier import SimpleConvolutionApplier


kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

gaussian_kernel = cv2.getGaussianKernel(3, 0)
gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)

# create a new OpenCVConvolutionApplier
opencv_convolution_applier = OpenCVConvolutionApplier(kernel)

# create a new SimpleConvolutionApplier
simple_convolution_applier = SimpleConvolutionApplier(kernel)

# apply the kernel to an image using OpenCV
opencv_convolution_applier.apply('images/marbles.bmp', 'outputs/marbles_opencv.bmp')

# apply the kernel to an image using the simple convolution applier
simple_convolution_applier.apply('images/marbles.bmp', 'outputs/marbles_simple.bmp')