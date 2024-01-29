import cv2
from convolution_appliers.abstract_convolution_applier import AbstractConvolutionApplier
from time import time

class OpenCVConvolutionApplier(AbstractConvolutionApplier):
    """An OpenCV implementation of a kernel convolution, decently fast"""

    def apply(self, image_path : str, output_path : str):
        """Apply a kernel filter to an image, returns the time it took to apply the filter"""
        # Read the image
        input_image = cv2.imread(image_path)

        # Measure the time it takes to apply the filter
        start_time = time()

        # Apply the filter to the image
        output_image = cv2.filter2D(input_image, -1, self.kernel)

        end_time = time()

        # Save the output image
        cv2.imwrite(output_path, output_image)

        return end_time - start_time