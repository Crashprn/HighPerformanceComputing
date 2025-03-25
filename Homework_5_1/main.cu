#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>

__global__
void grey_scale_kernel(unsigned char* greyImage, unsigned char* rgbImage, int image_width, int image_height)
{
    // Calculate the row and column of the current thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Make sure we do not go outside the image bounds
    if (col < image_width && row < image_height)
    {
        int g_index = row * image_width + col;
        int rgb_index = g_index * 3;
        // Calculate the grey value
        greyImage[g_index] = 0.21 * rgbImage[rgb_index] + 0.72 * rgbImage[rgb_index + 1] + 0.07 * rgbImage[rgb_index + 2];
    }
}

__host__
void read_image_file(std::string filename, char* buffer, int size)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        file.read(buffer, size);
        file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }
}

void write_image_file(std::string filename, char* buffer, int size)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        file.write(buffer, size);
        file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }
}

int main()
{
    // Set image properties
    int image_width = 1024;
    int image_height = 1024;
    int image_channels = 3;
    int image_size = image_width * image_height * image_channels;

    // Read image file
    char* rgb_image = new char[image_size];
    read_image_file("gc_conv_1024x1024.raw", rgb_image, image_size);

    // Allocate memory for grey image
    char* grey_image = new char[image_width * image_height];

    // Create device pointers
    unsigned char* d_rgb_image, *d_grey_image;

    // Copy data to device
    cudaMalloc((void**)&d_rgb_image, image_size * sizeof(unsigned char));
    cudaMemcpy(d_rgb_image, rgb_image, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_grey_image, image_width * image_height * sizeof(unsigned char));
    cudaMemcpy(d_grey_image, grey_image, image_width * image_height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 DimGrid(ceil(image_width / 16.0), ceil(image_height/ 16.0), 1);
    dim3 DimBlock(16, 16, 1);

    // Launch kernel
    grey_scale_kernel<<<DimGrid, DimBlock>>>(d_grey_image, d_rgb_image, image_width, image_height);

    // Copy data back to host
    cudaDeviceSynchronize();
    cudaMemcpy(grey_image, d_grey_image, image_width * image_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_rgb_image);
    cudaFree(d_grey_image);

    // Write grey image file
    write_image_file("gc_conv_1024x1024_grey.raw", grey_image, image_width * image_height);
}