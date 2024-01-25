import cv2
from abstract_convolution_applier import AbstractConvolutionApplier


class OpenCVConvolutionApplier(AbstractConvolutionApplier):
    """An OpenCV implementation of a kernel convolution, decently fast"""

    def apply(self, image_path : str, output_path : str):
        """Apply a kernel filter to an image"""
        # Read the image
        input_image = cv2.imread(image_path)

        # Apply the filter to the image
        output_image = cv2.filter2D(input_image, -1, self.kernel)

        # Save the output image
        cv2.imwrite(output_path, output_image)