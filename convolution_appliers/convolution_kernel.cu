#include <stdio.h>
__global__ void convolve(const unsigned char *input_image, unsigned char *output_image,
                                          const float * __restrict__ kernel, int width, int height,
                                          int channels, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int padding = max(1, kernel_size / 2);

    // Make sure we don't go out of bounds
    if (x >= padding && y >= padding && x < (width - padding) && y < (height - padding)) {
        for (int ch = 0; ch < channels; ch++) {
            float channel_sum = 0.0f;
            for (int ky = -padding; ky <= padding; ky++) {
                for (int kx = -padding; kx <= padding; kx++) {
                    float kernel_value = kernel[(ky + padding) * kernel_size + (kx + padding)];
                    channel_sum += input_image[((y + ky) * width + (x + kx)) * channels + ch] * kernel_value;
                }
            }
            output_image[(y * width + x) * channels + ch] = min(max(int(channel_sum), 0), 255);
        }
    }
}