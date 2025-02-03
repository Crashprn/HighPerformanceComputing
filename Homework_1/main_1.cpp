#include <vector>
#include <iostream>
#include <cmath>


void core_function(int n, int p, int num_tasks, int core_num, int* time_ptr, std::vector<int>& index_arr)
{
    int my_sum = 0;

    for (auto i = 0; i < num_tasks; ++i)
    {
        //int index = n - (p*(i+1) - ((i + core_num) % p));
        int index = n - (p * i + core_num) -1;
        if (index < 0) break;

        index_arr.push_back(index);
        *time_ptr += index+1;
        my_sum += index;
    }
}

int main()
{
    int n = 23;
    int p = 4;
    int num_tasks = static_cast<int>(std::ceil(static_cast<float>(n) / static_cast<float>(p)));

    int times[p] = {0};
    std::vector<std::vector<int>> index_arr;

    for (auto i = 0; i < p; ++i)
    {
        index_arr.push_back(std::vector<int>());
    }

    for (auto i = 0; i < p; ++i)
    {
        core_function(n, p, num_tasks, i, &times[i], index_arr[i]);
    }

    for (auto i = 0; i < p; ++i)
    {
        std::cout << "Core " << i << " time: " << times[i] << std::endl;
    }

    for (auto i = 0; i < p; ++i)
    {
        std::cout << "Core " << i << " index: ";
        for (auto j = 0; j < index_arr[i].size(); ++j)
        {
            std::cout << index_arr[i][j] << " ";
        }
        std::cout << std::endl;

    }

    return 0;
}