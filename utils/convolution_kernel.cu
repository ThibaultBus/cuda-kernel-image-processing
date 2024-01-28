__global__ void convolve(const unsigned char *input_image, unsigned char *output_image,
                                          const float * __restrict__ kernel, int width, int height,
                                          int channels, int kernel_size) {
    // Calculate the pixel's location
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // half of the kernel size
    int padding = max(1, kernel_size / 2);

    // Make sure we don't go out of bounds
    if (x >= padding && y >= padding && x < (width - padding) && y < (height - padding)) {
        // For each color channel
        for (int ch = 0; ch < channels; ch++) {
            float channel_sum = 0.0f;
            for (int ky = -padding; ky <= padding; ky++) {
                for (int kx = -padding; kx <= padding; kx++) {
                    // Get the kernel value at location (kx, ky)
                    float kernel_value = kernel[(ky + padding) * kernel_size + (kx + padding)];
                    // Get the image value at location (x + kx, y + ky) if given color channel and multiply by the kernel value
                    channel_sum += input_image[((y + ky) * width + (x + kx)) * channels + ch] * kernel_value;
                }
            }
            // Set the output image value at location (x, y) for given color channel, and clamp to [0, 255]
            output_image[(y * width + x) * channels + ch] = min(max(int(channel_sum), 0), 255);
        }
    }
}