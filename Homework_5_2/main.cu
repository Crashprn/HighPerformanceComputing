#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>

// Global Memory Transpose
__global__
void transpose_image_d(unsigned char* d_image, unsigned char* d_transposed_image, int width, int height, int channels)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < height)
    {
        int index_transposed = row * height + col;
        int index = col * width + row;
        for (int k = 0; k < channels; k++)
        {
            d_transposed_image[index_transposed * channels + k] = d_image[index * channels + k];
        }
    }
}

__host__
void transpose_image_h(char* image, char* transposed_image, int width, int height, int channels)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int index = i * width + j;
            int transposed_index = j * height + i;
            for (int k = 0; k < channels; k++)
            {
                transposed_image[transposed_index * channels + k] = image[index * channels + k];
            }
        }
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

__host__
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

__host__
bool compare_images(char* image1, char* image2, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (image1[i] != image2[i])
        {
            return false;
        }
    }
    return true;
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

    // Allocate memory for transposed images
    char* transposed_image_cpu = new char[image_width * image_height * image_channels];
    char* transposed_image_gpu = new char[image_width * image_height * image_channels];

    // Allocate memory on device
    unsigned char* d_rgb_image, *d_transposed_image;
    cudaMalloc((void**) &d_rgb_image, image_size* sizeof(char));
    cudaMalloc((void**) &d_transposed_image, image_size* sizeof(char));

    // Copy data to device
    cudaMemcpy(d_rgb_image, rgb_image, image_size* sizeof(char), cudaMemcpyHostToDevice);

    dim3 DimBlock(32, 32, 1);
    dim3 DimGrid(ceil(image_height / 32.0), ceil(image_width / 32.0), 1);

    // Execute kernel
    transpose_image_d<<<DimGrid, DimBlock>>>(d_rgb_image, d_transposed_image, image_width, image_height, image_channels);

    // Copy data back to host
    cudaDeviceSynchronize();
    cudaMemcpy(transposed_image_gpu, d_transposed_image, image_size* sizeof(char), cudaMemcpyDeviceToHost);

    // CPU transpose
    transpose_image_h(rgb_image, transposed_image_cpu, image_width, image_height, image_channels);

    // Compare images
    std::cout << "Images are equal: " << compare_images(transposed_image_cpu, transposed_image_gpu, image_size) << std::endl;

    // Write images to file
    write_image_file("transposed_image_cpu.raw", transposed_image_cpu, image_size);
    write_image_file("transposed_image_gpu.raw", transposed_image_gpu, image_size);
}