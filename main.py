import cv2
import numpy as np

from convolution_appliers.opencv_convolution_applier import OpenCVConvolutionApplier
from convolution_appliers.cuda_convolution_applier import CudaConvolutionApplier
from convolution_appliers.simple_convolution_applier import SimpleConvolutionApplier

gaussian_kernel = cv2.getGaussianKernel(3, 0)
gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)

kernel = gaussian_kernel_2d

#kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

# ensure the kernel is square and has an odd size
assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1

# create the convolution appliers
opencv_convolution_applier = OpenCVConvolutionApplier(kernel)
cuda_convolution_applier = CudaConvolutionApplier(kernel)
simple_convolution_applier = SimpleConvolutionApplier(kernel)

def get_output_path(input_path, applier):
    return input_path.replace("images", "outputs").replace(".", f"_{applier}.")

image_path = "images/starry_night.jpg"

# benchmark the OpenCVConvolutionApplier
opencv_time = opencv_convolution_applier.benchmark(image_path, get_output_path(image_path, "opencv"), iterations=5)

# benchmark the CudaConvolutionApplier
cuda_time = cuda_convolution_applier.benchmark(image_path, get_output_path(image_path, "cuda"), iterations=5)

# benchmark the SimpleConvolutionApplier, but only once because it's too slow
#simple_time = simple_convolution_applier.benchmark(image_path, get_output_path(image_path, "simple"), iterations=1)

# print the results in milliseconds
print(f"OpenCV: {opencv_time * 1000} ms")
print(f"CUDA: {cuda_time * 1000} ms")
#print(f"Simple: {simple_time * 1000} ms")

print("\n\n")

# How much faster is the CUDA implementation than the OpenCV implementation?
print(f"CUDA is {opencv_time // cuda_time} times faster than OpenCV")

# How much faster is the CUDA implementation than the simple implementation?
#print(f"CUDA is {simple_time // cuda_time} times faster than the simple implementation")


