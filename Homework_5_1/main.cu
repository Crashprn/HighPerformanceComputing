#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>

__global__
void grey_scale_kernel(unsigned char* greyImage, unsigned char* rgbImage, int image_width, int image_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < image_width && row < image_height)
    {
        int g_index = row * image_width + col;
        int rgb_index = g_index * 3;
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
    int image_width = 1024;
    int image_height = 1024;
    int image_channels = 3;
    int image_size = image_width * image_height * image_channels;


    char* rgb_image = new char[image_size];
    read_image_file("gc_conv_1024x1024.raw", rgb_image, image_size);

    char* grey_image = new char[image_width * image_height];

    unsigned char* d_rgb_image, *d_grey_image;
    cudaMalloc((void**)&d_rgb_image, image_size * sizeof(unsigned char));
    cudaMemcpy(d_rgb_image, rgb_image, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_grey_image, image_width * image_height * sizeof(unsigned char));
    cudaMemcpy(d_grey_image, grey_image, image_width * image_height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 DimGrid(ceil(image_width / 16.0), ceil(image_height/ 16.0), 1);
    dim3 DimBlock(16, 16, 1);
    grey_scale_kernel<<<DimGrid, DimBlock>>>(d_grey_image, d_rgb_image, image_width, image_height);

    cudaDeviceSynchronize();
    cudaMemcpy(grey_image, d_grey_image, image_width * image_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_rgb_image);
    cudaFree(d_grey_image);

    write_image_file("gc_conv_1024x1024_grey.raw", grey_image, image_width * image_height);
}