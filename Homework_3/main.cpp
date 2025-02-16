#include <thread>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

int* data_list;
int list_size;
int thread_count;

void count_sort(int data[], int n);
void memcpy(int* dest, int* src, int n);
void print_list(int* data, int n);


int main(int argc, char* argv[])
{   
    // Check for command line arguments
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <num_threads>" << " <list size>" << std::endl;
        return 1;
    }

    // Parse command line arguments
    thread_count = std::stoi(argv[1]);
    list_size = std::stoi(argv[2]);

    // Initialize random number generator and data list
    srand(100);
    data_list = new int[list_size];
    for (int i = 0; i < list_size; i++)
    {
        data_list[i] = rand() % 100;
    }

    // Print the list before sorting
    std::cout << "original: ";
    print_list(data_list, list_size);
    
    // Sort the list and measure time taken
    auto start = std::chrono::high_resolution_clock::now();
    count_sort(data_list, list_size);
    auto end = std::chrono::high_resolution_clock::now();

    // Print the list after sorting
    std::cout << "sorted: ";
    print_list(data_list, list_size);

    // Calculate and print the time taken
    float elapsed = std::chrono::duration<float, std::milli>(end - start).count();
    //std::cout << "Time taken: " << elapsed << " ms" << std::endl;

    // Clean up
    delete[] data_list;
    return 0;
}

// Parallel count sort
void count_sort(int a[], int n)
{
    // Initialize variables and temporary array
    int i, j, count;
    int* temp = new int[n];

    // Parallelize the sorting process
    #pragma omp parallel num_threads(thread_count) default(none) private(i, j, count) shared(a, temp, n)
    {
    
    // Calculate the count for each element parrallelly
    #pragma omp for
    for (i = 0; i < n; i++)
    {
        count = 0;
        for (j = 0; j < n; j++)
        {
            if (a[j] < a[i]) count++;
            else if (a[j] == a[i] && j < i) count++;
        }
        temp[count] = a[i];
    }

    // Synchronize the threads so all threads have finished sorting
    #pragma omp barrier
    memcpy(a, temp, n);
    }

    // Clean up
    delete[] temp;
}


// Thread-safe memcpy
void memcpy(int* dest, int* src, int n)
{
    int my_rank, num_threads;

    #ifdef _OPENMP
    // If OpenMP is enabled, get the thread rank and number of threads
    my_rank = omp_get_thread_num();
    num_threads = omp_get_num_threads();
    #else
    // If OpenMP is not enabled, set default values
    my_rank = 0;
    num_threads = 1;
    #endif

    // Calculate the range for each thread
    int my_n = n / num_threads;
    int my_start = my_rank * my_n;
    int my_end = (my_rank + 1) * my_n;
    if (my_rank == num_threads - 1) my_end = n;

    // Copy the data for this thread's range
    for (int i = my_start; i < my_end; i++)
    {
        dest[i] = src[i];
    }
}

// Print the list
void print_list(int* data, int n)
{
    for (int i = 0; i < n; i++)
    {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

