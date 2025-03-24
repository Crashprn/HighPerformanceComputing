#include <iostream>
#include <mpi.h>

int BIN_COUNT;
float MIN_MEAS;
float MAX_MEAS;
int DATA_COUNT;
int* BIN_COUNTS;
float* BIN_MAXES;
const int MASTER_RANK = 0;

// Function prototypes
void create_data(float* data, int dataCount, float minMeas, float maxMeas);
int get_bin_idx(float value, int bin_count);
void histogram_calc(float* data, int data_count, int* bin_counts);
void print_result(int* bin_counts, float* bin_maxes);
void data_displ_per_rank(int* data_count, int* data_displ, int comm_size, int data_count_total);


int main(int argc, char *argv[])
{
    int my_rank, comm_size;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == MASTER_RANK)
    {
        if (argc < 5)
        {
            std::cout << "Usage: " << argv[0] << " <bin count> <min meas> <max meas> <data count>" << std::endl;
            MPI_Finalize();
            return 1;
        }
        // Getting command line arguments
        BIN_COUNT = std::stoi(argv[1]);
        MIN_MEAS = std::stof(argv[2]);
        MAX_MEAS = std::stof(argv[3]);
        DATA_COUNT = std::stoi(argv[4]);
    }

    // Broadcast the arguments to all other processes
    MPI_Bcast(&BIN_COUNT, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&MIN_MEAS, 1, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&MAX_MEAS, 1, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&DATA_COUNT, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

    // Creating the maximums and counts arrays
    BIN_MAXES = new float[BIN_COUNT];
    BIN_COUNTS = new int[BIN_COUNT];
    for (int i = 0; i < BIN_COUNT; i++) {
        BIN_COUNTS[i] = 0;
        BIN_MAXES[i] = MIN_MEAS + (i + 1) * ((MAX_MEAS - MIN_MEAS) / static_cast<float>(BIN_COUNT));
    }

    // Calculating the number of data points for each process
    int my_data_count = DATA_COUNT / comm_size;
    if (my_rank == comm_size - 1)
    {
        my_data_count += DATA_COUNT % comm_size;
    } 
    // Making buffer for each process
    float* my_data = new float[my_data_count];
    
    // Scatter the data from the master process to all other processes
    if (my_rank == MASTER_RANK)
    {
        // Creating the data array with random values
        float* data = new float[DATA_COUNT];
        create_data(data, DATA_COUNT, MIN_MEAS, MAX_MEAS);

        // Defining array to hold the number of data points and displ value for each process
        int* count_per_process = new int[comm_size];
        int* displacement = new int[comm_size];
        data_displ_per_rank(count_per_process, displacement, comm_size, DATA_COUNT);

        MPI_Scatterv(data, count_per_process, displacement, MPI_FLOAT, my_data, my_data_count, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);

        // Deleting the data array
        delete[] data;
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, my_data, my_data_count, MPI_FLOAT, MASTER_RANK, MPI_COMM_WORLD);
    }

    
    // Calculate the histogram for this process
    histogram_calc(my_data, my_data_count, BIN_COUNTS);

    // Reduce the histogram counts to the root process
    if (my_rank == MASTER_RANK)
    {
        int global_counts[BIN_COUNT] = {0};
        MPI_Reduce(BIN_COUNTS, global_counts, BIN_COUNT, MPI_INT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
        print_result(global_counts, BIN_MAXES);
    }
    else
    {
        MPI_Reduce(BIN_COUNTS, NULL, BIN_COUNT, MPI_INT, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
    }
            
    MPI_Finalize();
    return 0;
}

void create_data(float* data, int dataCount, float minMeas, float maxMeas)
{
    srand(100); // Seed the random number generator

    // Generate random data and store it in the data array
    for (int i = 0; i < dataCount; i++) {
        data[i] = minMeas + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxMeas - minMeas)));
    }

}

// Binary search to find the bin index for a given value
// Returns the index of the bin that the value belongs to
int get_bin_idx(float value, int bin_count)
{ 
    int low = 0;
    int high = bin_count - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (BIN_MAXES[mid] == value)
        {
            return mid;
        }
        else if (value < BIN_MAXES[mid])
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

// Increments the count for the bin that the value belongs to
void histogram_calc(float* data, int data_count, int* bin_counts)
{
    for (int i = 0; i < data_count; i++) {
        int bin = get_bin_idx(data[i], BIN_COUNT);
        bin_counts[bin]++;
    }
}

// Print the result of the program
void print_result(int* bin_counts, float* bin_maxes)
{   
    std::cout << "bin_maxes:  ";
    for (int i = 0; i < BIN_COUNT; i++) {
        std::cout << bin_maxes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "bin_counts: ";
    for (int i = 0; i < BIN_COUNT; i++) {
        std::cout << bin_counts[i] << " ";
    }
    std::cout << std::endl;
}

// Calculates the number of data points for each process and the displ value
// for each process. Gives all remaining data points to the last process.
void data_displ_per_rank(int* data_count, int* data_displ, int comm_size, int data_count_total)
{
    int data_per_process = data_count_total / comm_size;
    for (int i = 0; i < comm_size; i++) 
    {
        data_displ[i] = i * data_per_process;
        data_count[i] = data_per_process;
        if (i == comm_size - 1) 
        {
            data_count[i] = data_count_total - (data_per_process * i);
        }
    }
}
