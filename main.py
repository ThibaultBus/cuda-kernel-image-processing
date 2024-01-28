import cv2
import numpy as np

from convolution_appliers.opencv_convolution_applier import OpenCVConvolutionApplier
from convolution_appliers.cuda_convolution_applier import CudaConvolutionApplier
from convolution_appliers.simple_convolution_applier import SimpleConvolutionApplier

from utils.image_resizer_steps import incremental_image_resize
from utils.get_output_path import get_output_path

import csv

gaussian_kernel = cv2.getGaussianKernel(9, 0)
gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)

kernel = gaussian_kernel_2d

#kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

# ensure the kernel is square and has an odd size
assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1

# create the convolution appliers
opencv_convolution_applier = OpenCVConvolutionApplier(kernel)
cuda_convolution_applier = CudaConvolutionApplier(kernel)
simple_convolution_applier = SimpleConvolutionApplier(kernel)

images_paths = incremental_image_resize("images/starry_night.jpg", "outputs", 100, 4000, 100)

# open a CSV file to write the results
with open("results.csv", "w") as csv_file:
    csv_writer = csv.writer(csv_file)

    # write the header
    csv_writer.writerow(["image_path", "opencv_time", "cuda_time", "simple_time"])

    last_simple_time = 0

    # write the results for each image
    for image_path in images_paths:
        # benchmark the OpenCVConvolutionApplier
        opencv_time = opencv_convolution_applier.benchmark(image_path, get_output_path(image_path, "opencv"), iterations=5)

        # benchmark the CudaConvolutionApplier
        cuda_time = cuda_convolution_applier.benchmark(image_path, get_output_path(image_path, "cuda"), iterations=5)

        # benchmark the SimpleConvolutionApplier, but stops when it starts to take more than 1 seconds as it would be unplotable on the graph
        simple_time = 0
        if last_simple_time < 1.:
            simple_time = simple_convolution_applier.benchmark(image_path, get_output_path(image_path, "simple"), iterations=5)
            last_simple_time = simple_time
        else:
            simple_time = 0
        
        # print the results in milliseconds
        print(f"OpenCV: {opencv_time * 1000} ms")
        print(f"CUDA: {cuda_time * 1000} ms")
        print(f"Simple: {simple_time * 1000} ms")

        print("\n\n")

        # How much faster is the CUDA implementation than the OpenCV implementation?
        print(f"CUDA is {opencv_time // cuda_time} times faster than OpenCV")

        # How much faster is the CUDA implementation than the simple implementation?
        print(f"CUDA is {simple_time // cuda_time} times faster than the simple implementation")

        # write the results to the CSV file
        csv_writer.writerow([image_path, opencv_time, cuda_time, simple_time])