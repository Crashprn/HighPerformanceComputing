/*
TO RUN THIS CODE BECAUSE SEMAPHORES ARE NOT PART OF <C++20 STANDARD LIBRARY ON CHPC:

module load gcc/13.1.0
g++ -std=c++20 -pthread main.cpp -o main

*/



#include <thread>
#include <iostream>
#include <cmath>
#include <tuple>
#include <functional>
#include <semaphore>
#include <vector>
#include <format>
#include <chrono>

int NUM_THREADS;
int BIN_COUNT;
float* DATA;
float MIN_MEAS;
float MAX_MEAS;
int DATA_COUNT;
int **BIN_COUNTS;
float* BIN_MAXES;
std::vector<std::unique_ptr<std::binary_semaphore>> GLOBAL_SEMAPHORES;
std::vector<std::unique_ptr<std::binary_semaphore>> TREE_SEMAPHORES;


void print_result(std::string name, float* times);
void reset_bin_counts();
int get_bin_idx(float value, int bin_count);
void add_bin_counts(int target, int source);
void histogram_thread(std::tuple<int, int, float*, std::function<void(int)>> args);
void global_sum(int thread_id);
void tree_sum(int thread_id);

int main(int argc, char *argv[]) {
    // Checking for correct number of arguments
    if (argc < 6) {
        std::cout << "Usage: " << argv[0] << " <number of threads> <bin count> <min meas> <max meas> <data count>" << std::endl;
        return 1;
    }

    // Getting command line arguments
    NUM_THREADS = std::stoi(argv[1]);
    BIN_COUNT = std::stoi(argv[2]);
    MIN_MEAS = std::stof(argv[3]);
    MAX_MEAS = std::stof(argv[4]);
    DATA_COUNT = std::stoi(argv[5]);
    int data_per_thread = std::ceil(static_cast<float>(DATA_COUNT) / static_cast<float>(NUM_THREADS));

    // Seeding the random number generator
    srand(100);

    // Creating the maximums and counts arrays
    float bin_inc = (MAX_MEAS - MIN_MEAS) / static_cast<float>(BIN_COUNT);
    BIN_MAXES = new float[BIN_COUNT];
    BIN_COUNTS = new int*[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        BIN_COUNTS[i] = new int[BIN_COUNT];
    }
    for (int i = 0; i < BIN_COUNT; i++) {
        BIN_MAXES[i] = MIN_MEAS + (i + 1) * bin_inc;
    }

    // Creating the data array with random values
    DATA = new float[DATA_COUNT];
    for (int i = 0; i < DATA_COUNT; i++) {
        DATA[i] = MIN_MEAS + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (MAX_MEAS - MIN_MEAS)));
    }

    // Initializing list of threads 
    std::thread threads[NUM_THREADS];

    // Global Sum
    float times_global[NUM_THREADS] = {0};

    // Allocating Semaphores for global sum
    for (int i = 0; i < NUM_THREADS; i++) {
        GLOBAL_SEMAPHORES.emplace_back(std::make_unique<std::binary_semaphore>(0));
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Creating the threads and running the histogram
    for (int i = 0; i < NUM_THREADS ; i++) {
        auto args = std::tuple<int, int, float*, std::function<void(int)>>(i, data_per_thread, &times_global[i], global_sum);
        threads[i] = std::thread(histogram_thread, args);
    }

    // Joining the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float, std::milli>(end - start).count();

    // Printing the results
    print_result("Global Sum", times_global);
    //std::cout << "Elapsed time: " << elapsed << std::endl;

    // Tree Sum

    reset_bin_counts();
    // Allocating Semaphores
    for (int i = 0; i < NUM_THREADS; i++) {
       TREE_SEMAPHORES.emplace_back(std::make_unique<std::binary_semaphore>(0));
    }

    float times_tree[NUM_THREADS] = {0};

    start = std::chrono::high_resolution_clock::now();
    // Creating the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        auto args = std::tuple<int, int, float*, std::function<void(int)>>(i, data_per_thread, &times_tree[i], tree_sum);
        threads[i] = std::thread(histogram_thread, args);
    }

    // Joining the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration<float, std::milli>(end - start).count();


    // Printing the results, max thread time, and elapsed time
    print_result("Tree Structured Sum", times_tree);
    //std::cout << "Elapsed Time: " << elapsed << std::endl;

    // Freeing the allocated memory
    delete[] DATA;
    delete[] BIN_MAXES;

}

// Resets the bin counts to 0
void reset_bin_counts()
{
    for (int i = 0; i < NUM_THREADS; i++) {
        for (int j = 0; j < BIN_COUNT; j++) {
            BIN_COUNTS[i][j] = 0;
        }
    }
}

void print_result(std::string name, float* times)
{

    std::cout << name << std::endl;
    std::cout << "bin_maxes:  ";
    for (int i = 0; i < BIN_COUNT; i++) {
        std::cout << std::format("{:3.2f} ", BIN_MAXES[i]);
    }
    std::cout << std::endl;
    int final_count = 0;
    std::cout << "bin_counts: ";
    for (int i = 0; i < BIN_COUNT; i++) {
        std::cout << std::format("{:4d} ", BIN_COUNTS[0][i]);
	final_count += BIN_COUNTS[0][i];
    }
    std::cout << std::endl;
    // std::cout << "Bin count matches data: " << (final_count == DATA_COUNT ? "true" : "false") << std::endl;
    // std::cout << "Thread Max Time: " << *std::max_element(times, times + NUM_THREADS) << std::endl;
}


// Simple global sum for the bin counts
void global_sum(int thread_id)
{
    if (thread_id == 0) 
    {
        for (int i = 1; i < NUM_THREADS; i++) {
            GLOBAL_SEMAPHORES[i]->acquire();
            add_bin_counts(0, i);
        }
    }
    else
    {
        GLOBAL_SEMAPHORES[thread_id]->release();
    }
}

// Tree structured sum for the bin counts
void tree_sum(int thread_id)
{
    auto current_idx = thread_id;
    auto thread_offset = 1;

    // If a thread receives a sum, it receives the +1, +2, +4, ..., sums until it is odd
    while (current_idx % 2 == 0) {
        int receive_idx = thread_id + thread_offset;

        // Ensure we don't go out of bounds
        if (receive_idx >= NUM_THREADS) {
            break;
        }
        // Wait for receive thread to finish
        TREE_SEMAPHORES[receive_idx]->acquire();

        add_bin_counts(thread_id, receive_idx);

        // divide by 2 to get the parent and multiply by 2 to get adjacent child
        current_idx /= 2;
        thread_offset *= 2;
    }

    // Finally, release the semaphore for the parent thread
    // Except for 0 because it is the root
    if (thread_id != 0)
    {
        TREE_SEMAPHORES[thread_id]->release();
    }
}

// Adds the bin counts of one source thread to the target thread
void add_bin_counts(int target, int source)
{
    for (int j = 0; j < BIN_COUNT; j++) {
        BIN_COUNTS[target][j] += BIN_COUNTS[source][j];
    }
}

// Thread function to calculate the histogram
void histogram_thread(std::tuple<int, int, float*, std::function<void(int)>> args)
{
    // Unpacking the arguments and setting up the thread
    int thread_id = std::get<0>(args);
    int data_per_thread = std::get<1>(args);
    float* time = std::get<2>(args);
    std::function<void(int)> sum_function = std::get<3>(args);
    int* my_bin_count = BIN_COUNTS[thread_id];
    int my_start = thread_id * data_per_thread;
    int my_end = (thread_id + 1) * data_per_thread;

    auto start = std::chrono::high_resolution_clock::now();

    // Calculate the histogram for this thread
    for (int i = my_start; i < my_end; i++) {
        if (i >= DATA_COUNT) {
            break;
        }
        int bin = get_bin_idx(DATA[i], BIN_COUNT);
        my_bin_count[bin]++;
    }

    // Call the function to sum together bins
    sum_function(thread_id);

    // Calculate the time taken for this thread and assign it to the time pointer
    auto end = std::chrono::high_resolution_clock::now();
    *time = std::chrono::duration<float, std::milli>(end - start).count();
}

// Simple binary search to find the bin for a given value
int get_bin_idx(float value, int bin_count)
{ 
    int low = 0;
    int high = bin_count - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (value < BIN_MAXES[mid])
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
    return low;
}
