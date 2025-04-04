#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>
#include <math.h>

#define TILE_WIDTH 32

// Global Memory Transpose
__global__
void transpose_image_reg(unsigned char* d_image, unsigned char* d_transposed_image, int width, int height, int channels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width)
    {
        int t_index = col * height + row;
        int r_index = row * width + col;

        for (int k = 0; k < channels; k++)
        {
            d_transposed_image[t_index * channels + k] = d_image[r_index * channels + k];
        }
    }
}

__global__
void transpose_image_tile(unsigned char* d_image, unsigned char* d_transposed_image, int width, int height, int channels)
{
    __shared__ unsigned char tile[TILE_WIDTH][TILE_WIDTH][3]; // Assuming 3 channels (RGB)
    
    // Calculate row and column indices
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    if (row < height && col < width)
    {

        // load data into shared memory
        int r_index = row * width + col;
        for (int k = 0; k < channels; k++)
        {
            tile[threadIdx.x][threadIdx.y][k] = d_image[r_index * channels + k];
        }

        __syncthreads(); // Ensure all threads have loaded their data

        // Write transposed data to global memory
        int t_col = blockIdx.y * TILE_WIDTH + threadIdx.x; // Transposed column index
        int t_row = blockIdx.x * TILE_WIDTH + threadIdx.y; // Transposed row index
        int index_transposed = t_row * height + t_col;

        for (int k = 0; k < channels; k++)
        {
            d_transposed_image[index_transposed * channels + k] = tile[threadIdx.y][threadIdx.x][k];
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
    char* transposed_image_gpu_reg = new char[image_width * image_height * image_channels];
    char* transposed_image_gpu_tile = new char[image_width * image_height * image_channels];

    // Allocate memory on device
    unsigned char* d_rgb_image, *d_transposed_image_reg, *d_transposed_image_tile;
    cudaMalloc((void**) &d_rgb_image, image_size* sizeof(char));
    cudaMalloc((void**) &d_transposed_image_reg, image_size* sizeof(char));
    cudaMalloc((void**) &d_transposed_image_tile, image_size* sizeof(char));

    // Copy data to device
    cudaMemcpy(d_rgb_image, rgb_image, image_size* sizeof(char), cudaMemcpyHostToDevice);

    // Creating grid and block dimensions
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 DimGrid(ceil(image_height * 1.0 / TILE_WIDTH), ceil(image_width * 1.0 / TILE_WIDTH), 1);

    // Creating timing events
    cudaEvent_t start, stop;
    float elapsed_time_reg, elapsed_time_tile;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execute global kernel
    cudaEventRecord(start, 0);
    transpose_image_reg<<<DimGrid, DimBlock>>>(d_rgb_image, d_transposed_image_reg, image_width, image_height, image_channels);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_reg, start, stop);

    // Execute tiled kernel
    cudaEventRecord(start, 0);
    transpose_image_tile<<<DimGrid, DimBlock>>>(d_rgb_image, d_transposed_image_tile, image_width, image_height, image_channels);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_tile, start, stop);

    // Copy data back to host
    cudaMemcpy(transposed_image_gpu_reg, d_transposed_image_reg, image_size* sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(transposed_image_gpu_tile, d_transposed_image_tile, image_size* sizeof(char), cudaMemcpyDeviceToHost);

    // CPU transpose
    transpose_image_h(rgb_image, transposed_image_cpu, image_width, image_height, image_channels);

    // Compare images
    std::cout << "Images are equal regular: " << compare_images(transposed_image_cpu, transposed_image_gpu_reg, image_size) << std::endl;
    std::cout << "Images are equal tiled: " << compare_images(transposed_image_cpu, transposed_image_gpu_tile, image_size) << std::endl;

    // Calculating bandwidth
    int total_bytes = (image_size * 2); // Total bytes transferred (input + output)
    float total_Gbytes = total_bytes / pow(10.0, 9.0); // Convert bytes to Gbytes

    float bandwidth_reg = total_Gbytes / (elapsed_time_reg / 1000.0); // Gbytes per second
    float bandwidth_tile = total_Gbytes / (elapsed_time_tile / 1000.0); // Gbytes per second

    std::cout << "Elapsed time for regular transpose: " << elapsed_time_reg << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_reg << std::endl;
    std::cout << "Elapsed time for tiled transpose: " << elapsed_time_tile << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_tile << std::endl;

    // Write images to file
    write_image_file("transposed_image_cpu.raw", transposed_image_cpu, image_size);
    write_image_file("transposed_image_gpu_reg.raw", transposed_image_gpu_reg, image_size);
    write_image_file("transposed_image_gpu_tile.raw", transposed_image_gpu_tile, image_size);
}