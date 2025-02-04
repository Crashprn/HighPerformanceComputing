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
    /*
    // Setting defaults
    NUM_THREADS = 1;
    BIN_COUNT = 10;
    MIN_MEAS = 0.0;
    MAX_MEAS = 5.0;
    DATA_COUNT = 100;
    */

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

    int data_per_thread = std::ceil(static_cast<float>(DATA_COUNT) / static_cast<float>(NUM_THREADS));

    
    
    std::thread threads[NUM_THREADS];

    // Global Sum
    float times_global[NUM_THREADS] = {0};

    // Allocating Semaphores
    for (int i = 0; i < NUM_THREADS; i++) {
        GLOBAL_SEMAPHORES.emplace_back(std::make_unique<std::binary_semaphore>(0));
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Creating the threads
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
    std::cout << "Elapsed time: " << elapsed << std::endl;

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


    // Printing the results
    print_result("Tree Structured Sum", times_tree);
    std::cout << "Elapsed Time: " << elapsed << std::endl;

    delete[] DATA;
    delete[] BIN_MAXES;

}

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
    std::cout << "bin_counts: ";
    for (int i = 0; i < BIN_COUNT; i++) {
        std::cout << std::format("{:4d} ", BIN_COUNTS[0][i]);
    }
    std::cout << std::endl;
    for (auto i = 0; i < NUM_THREADS; i++) {
        std::cout << std::format("{:3.2f} ", times[i]);
    }
    std::cout << std::endl;

}

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

void tree_sum(int thread_id)
{

    auto current_idx = thread_id;
    auto thread_offset = 1;
    while (current_idx % 2 == 0) {
        int receive_idx = thread_id + thread_offset;
        if (receive_idx >= NUM_THREADS) {
            break;
        }
        TREE_SEMAPHORES[receive_idx]->acquire();

        add_bin_counts(thread_id, receive_idx);
        current_idx /= 2;
        thread_offset *= 2;
    }

    if (thread_id != 0)
    {
        TREE_SEMAPHORES[thread_id]->release();
    }
}

void add_bin_counts(int target, int source)
{
    for (int j = 0; j < BIN_COUNT; j++) {
        BIN_COUNTS[target][j] += BIN_COUNTS[source][j];
    }
}

void histogram_thread(std::tuple<int, int, float*, std::function<void(int)>> args)
{
    int thread_id = std::get<0>(args);
    int data_per_thread = std::get<1>(args);
    float* time = std::get<2>(args);
    std::function<void(int)> sum_function = std::get<3>(args);
    int* my_bin_count = BIN_COUNTS[thread_id];
    int my_start = thread_id * data_per_thread;
    int my_end = (thread_id + 1) * data_per_thread;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = my_start; i < my_end; i++) {
        if (i >= DATA_COUNT) {
            break;
        }
        int bin = get_bin_idx(DATA[i], BIN_COUNT);
        my_bin_count[bin]++;
    }
    sum_function(thread_id);
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
