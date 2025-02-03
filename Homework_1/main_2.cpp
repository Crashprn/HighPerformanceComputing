#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>



int MAX = 8;
int CACHE_LINE = 4;

bool tuple_equal(std::tuple<int,int> t1, int i, int j)
{
    return std::get<0>(t1) == i and std::get<1>(t1) == j;
}

void update_cache(int i, int j, std::vector<std::vector<std::tuple<int,int>>>& cache)
{
    bool in_cache = false;

    if (cache.size() > 0)
    {
        int pot_start = cache.size() - CACHE_LINE;
        int start = (pot_start < 0) ? 0 : (cache.size() - CACHE_LINE);
        for (auto k = start; k < cache.size(); ++k)
        {
            if (std::find_if(cache[k].begin(), cache[k].end(), [i,j](const std::tuple<int,int> e){return tuple_equal(e,i,j);}) != cache[k].end())
            {
                in_cache = true;
            }
        }

    }
    
    if (!in_cache)
    {
        std::vector<std::tuple<int,int>> new_cache_line;
        for (auto k = 0; k < CACHE_LINE; ++k)
        {
            if (j + k >= MAX)
            {
                i = (i + 1) % MAX;
            } 
            new_cache_line.push_back(std::make_tuple(i,(j + k) % MAX));            
        }
        cache.push_back(new_cache_line);
    }

}

void print_cache(std::vector<std::vector<std::tuple<int,int>>>& cache)
{
    for (auto i = 0; i < cache.size(); ++i)
    {
        std::cout << "Cache line " << i % CACHE_LINE << ": ";
        for (auto j = 0; j < cache[i].size(); ++j)
        {
            std::cout << "A[" << std::get<0>(cache[i][j]) << "][" << std::get<1>(cache[i][j]) << "] ";
        }
        std::cout << std::endl;
    }
}


int main()
{
    int A[MAX][MAX] = {0};
    std::vector<std::vector<std::tuple<int,int>>> cache_1;
    std::vector<std::vector<std::tuple<int,int>>> cache_2;


    for (auto i = 0; i < MAX; ++i)
    {
        for (auto j = 0; j < MAX; ++j)
        {
            update_cache(i,j,cache_1);
        }
    }

    for (auto j = 0; j < MAX; ++j)
    {
        for (auto i = 0; i < MAX; ++i)
        {
            update_cache(i,j,cache_2);
        }
    }

    print_cache(cache_1);
    std::cout << "----------------" << std::endl;
    print_cache(cache_2);

    std::cout << "Cache 1 Reads: " << cache_1.size() << std::endl;
    std::cout << "Cache 2 Reads: " << cache_2.size() << std::endl;

    return 0;
}