#include <algorithm>
#include <vector>

#include "wrapper.hpp"

int main(void) {
    std::vector<std::vector<std::size_t>> E2C2V{{0, 1, 4, 6},
        {0, 4, 1, 3},
        {0, 3, 4, 2},
        {1, 2, 5, 7},
        {1, 5, 2, 4},
        {1, 4, 0, 5},
        {2, 0, 3, 8},
        {2, 3, 0, 5},
        {2, 5, 1, 3},
        {3, 4, 0, 7},
        {3, 7, 4, 6},
        {3, 6, 7, 5},
        {4, 5, 1, 8},
        {4, 8, 5, 7},
        {4, 7, 3, 8},
        {5, 3, 2, 6},
        {5, 6, 3, 8},
        {5, 8, 4, 6},
        {6, 7, 3, 1},
        {6, 1, 7, 0},
        {6, 0, 1, 8},
        {7, 8, 4, 2},
        {7, 2, 8, 1},
        {7, 1, 6, 2},
        {8, 6, 5, 0},
        {8, 0, 6, 2},
        {8, 2, 7, 0}};
    std::vector<std::vector<std::size_t>> E2ECV{{0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 10, 11},
        {12, 13, 14, 15},
        {16, 17, 18, 19},
        {20, 21, 22, 23},
        {24, 25, 26, 27},
        {28, 29, 30, 31},
        {32, 33, 34, 35},
        {36, 37, 38, 39},
        {40, 41, 42, 43},
        {44, 45, 46, 47},
        {48, 49, 50, 51},
        {52, 53, 54, 55},
        {56, 57, 58, 59},
        {60, 61, 62, 63},
        {64, 65, 66, 67},
        {68, 69, 70, 71},
        {72, 73, 74, 75},
        {76, 77, 78, 79},
        {80, 81, 82, 83},
        {84, 85, 86, 87},
        {88, 89, 90, 91},
        {92, 93, 94, 95},
        {96, 97, 98, 99},
        {100, 101, 102, 103},
        {104, 105, 106, 107}};
    auto nabla4_benchmark_unstructured_naive_runtimes =
        nabla4_benchmark_unstructured_naive(E2C2V, E2ECV, 18, 9, 27, 65, 4, 1, 0);
    std::sort(nabla4_benchmark_unstructured_naive_runtimes.begin(), nabla4_benchmark_unstructured_naive_runtimes.end());
    std::cout << "Unstructured naive nabla4 median runtime: "
              << nabla4_benchmark_unstructured_naive_runtimes[nabla4_benchmark_unstructured_naive_runtimes.size() / 2]
              << std::endl;
    std::vector<std::vector<std::size_t>> E2ECV_structured{{0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 11, 10}, // 2
        {12, 13, 14, 15},
        {16, 17, 18, 19},
        {20, 21, 22, 23},
        {24, 25, 26, 27},
        {28, 29, 30, 31},
        {32, 33, 34, 35},
        {36, 37, 39, 38}, // 9
        {40, 41, 42, 43},
        {44, 45, 47, 46}, // 11
        {48, 49, 51, 50}, // 12
        {52, 53, 54, 55},
        {56, 57, 58, 59},
        {60, 61, 63, 62}, // 15
        {64, 65, 66, 67},
        {68, 69, 70, 71},
        {72, 73, 75, 74}, // 18
        {76, 77, 78, 79},
        {80, 81, 83, 82}, // 20
        {84, 85, 87, 86}, // 21
        {88, 89, 90, 91},
        {92, 93, 94, 95},
        {96, 97, 99, 98}, // 24
        {100, 101, 102, 103},
        {104, 105, 106, 107}};
    auto nabla4_benchmark_structured_naive_runtimes =
        nabla4_benchmark_structured_naive(E2C2V, E2ECV_structured, 18, 9, 27, 65, 4, 1, 0);
    std::sort(nabla4_benchmark_structured_naive_runtimes.begin(), nabla4_benchmark_structured_naive_runtimes.end());
    std::cout << "Unstructured naive nabla4 median runtime: "
              << nabla4_benchmark_structured_naive_runtimes[nabla4_benchmark_unstructured_naive_runtimes.size() / 2]
              << std::endl;
}
