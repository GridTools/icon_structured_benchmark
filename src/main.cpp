#include <algorithm>
#include <vector>

#include "wrapper.hpp"

int main(void) {
    std::vector<std::array<std::size_t, 4>> E2C2V{{0, 1, 4, 6},
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
    std::vector<std::array<std::size_t, 4>> E2ECV{{0, 1, 2, 3},
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

    int repetitions = 101;
    int dry_runs = 1;

    auto nabla4_benchmark_unstructured_naive_runtimes =
        nabla4_benchmark_unstructured_naive(E2C2V, E2ECV, 18, 9, 27, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_unstructured_naive_runtimes.begin(), nabla4_benchmark_unstructured_naive_runtimes.end());
    std::cout << "Unstructured SimpleGrid naive nabla4 median runtime: "
              << nabla4_benchmark_unstructured_naive_runtimes[nabla4_benchmark_unstructured_naive_runtimes.size() / 2]
              << std::endl;
    auto nabla4_benchmark_unstructured_cpu_ifirst_runtimes =
        nabla4_benchmark_unstructured_cpu_ifirst(E2C2V, E2ECV, 18, 9, 27, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_unstructured_cpu_ifirst_runtimes.begin(),
        nabla4_benchmark_unstructured_cpu_ifirst_runtimes.end());
    std::cout
        << "Unstructured SimpleGrid cpu_ifirst nabla4 median runtime: "
        << nabla4_benchmark_unstructured_cpu_ifirst_runtimes[nabla4_benchmark_unstructured_cpu_ifirst_runtimes.size() /
                                                             2]
        << std::endl;
    auto nabla4_benchmark_unstructured_cpu_kfirst_runtimes =
        nabla4_benchmark_unstructured_cpu_kfirst(E2C2V, E2ECV, 18, 9, 27, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_unstructured_cpu_kfirst_runtimes.begin(),
        nabla4_benchmark_unstructured_cpu_kfirst_runtimes.end());
    std::cout
        << "Unstructured SimpleGrid cpu_kfirst nabla4 median runtime: "
        << nabla4_benchmark_unstructured_cpu_kfirst_runtimes[nabla4_benchmark_unstructured_cpu_kfirst_runtimes.size() /
                                                             2]
        << std::endl;

    auto nabla4_benchmark_structured_simple_naive_runtimes =
        nabla4_benchmark_structured_simple_naive(18, 9, 27, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_structured_simple_naive_runtimes.begin(),
        nabla4_benchmark_structured_simple_naive_runtimes.end());
    std::cout
        << "Structured SimpleGrid naive nabla4 median runtime: "
        << nabla4_benchmark_structured_simple_naive_runtimes[nabla4_benchmark_structured_simple_naive_runtimes.size() /
                                                             2]
        << std::endl;
    auto nabla4_benchmark_structured_simple_cpu_ifirst_runtimes =
        nabla4_benchmark_structured_simple_cpu_ifirst(18, 9, 27, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_structured_simple_cpu_ifirst_runtimes.begin(),
        nabla4_benchmark_structured_simple_cpu_ifirst_runtimes.end());
    std::cout << "Structured SimpleGrid cpu_ifirst nabla4 median runtime: "
              << nabla4_benchmark_structured_simple_cpu_ifirst_runtimes
                     [nabla4_benchmark_structured_simple_cpu_ifirst_runtimes.size() / 2]
              << std::endl;
    auto nabla4_benchmark_structured_simple_cpu_kfirst_runtimes =
        nabla4_benchmark_structured_simple_cpu_kfirst(18, 9, 27, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_structured_simple_cpu_kfirst_runtimes.begin(),
        nabla4_benchmark_structured_simple_cpu_kfirst_runtimes.end());
    std::cout << "Structured SimpleGrid cpu_kfirst nabla4 median runtime: "
              << nabla4_benchmark_structured_simple_cpu_kfirst_runtimes
                     [nabla4_benchmark_structured_simple_cpu_kfirst_runtimes.size() / 2]
              << std::endl;

    std::vector<std::array<std::size_t, 4>> E2C2V_torus_24576{{0, 4, 11, 1},
        {0, 1, 4, 9},
        {0, 9, 1, 8},
        {1, 5, 4, 2},
        {1, 2, 5, 10},
        {1, 10, 2, 9},
        {2, 6, 5, 3},
        {2, 3, 6, 11},
        {2, 11, 3, 10},
        {3, 7, 6, 8},
        {3, 8, 7, 4},
        {3, 4, 8, 11},
        {4, 8, 3, 5},
        {4, 5, 8, 1},
        {4, 1, 5, 0},
        {5, 9, 8, 6},
        {5, 6, 9, 2},
        {5, 2, 6, 1},
        {6, 10, 9, 7},
        {6, 7, 10, 3},
        {6, 3, 7, 2},
        {7, 11, 10, 0},
        {7, 0, 11, 8},
        {7, 8, 0, 3},
        {8, 0, 7, 9},
        {8, 9, 0, 5},
        {8, 5, 9, 4},
        {9, 1, 0, 10},
        {9, 10, 1, 6},
        {9, 6, 10, 5},
        {10, 2, 1, 11},
        {10, 11, 2, 7},
        {10, 7, 11, 6},
        {11, 3, 2, 4},
        {11, 4, 3, 0},
        {11, 0, 4, 7}};
    std::vector<std::array<std::size_t, 4>> E2ECV_torus_24576{{0, 1, 2, 3},
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
        {104, 105, 106, 107},
        {108, 109, 110, 111},
        {112, 113, 114, 115},
        {116, 117, 118, 119},
        {120, 121, 122, 123},
        {124, 125, 126, 127},
        {128, 129, 130, 131},
        {132, 133, 134, 135},
        {136, 137, 138, 139},
        {140, 141, 142, 143}};

    auto nabla4_benchmark_unstructured_torus_naive_runtimes = nabla4_benchmark_unstructured_naive(
        E2C2V_torus_24576, E2ECV_torus_24576, 24, 12, 36, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_unstructured_torus_naive_runtimes.begin(),
        nabla4_benchmark_unstructured_torus_naive_runtimes.end());
    std::cout << "Unstructured Torus 24576 naive nabla4 median runtime: "
              << nabla4_benchmark_unstructured_torus_naive_runtimes
                     [nabla4_benchmark_unstructured_torus_naive_runtimes.size() / 2]
              << std::endl;
    auto nabla4_benchmark_unstructured_torus_cpu_ifirst_runtimes = nabla4_benchmark_unstructured_cpu_ifirst(
        E2C2V_torus_24576, E2ECV_torus_24576, 24, 12, 36, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_unstructured_torus_cpu_ifirst_runtimes.begin(),
        nabla4_benchmark_unstructured_torus_cpu_ifirst_runtimes.end());
    std::cout << "Unstructured Torus 24576 cpu_ifirst nabla4 median runtime: "
              << nabla4_benchmark_unstructured_torus_cpu_ifirst_runtimes
                     [nabla4_benchmark_unstructured_torus_cpu_ifirst_runtimes.size() / 2]
              << std::endl;
    auto nabla4_benchmark_unstructured_torus_cpu_kfirst_runtimes = nabla4_benchmark_unstructured_cpu_kfirst(
        E2C2V_torus_24576, E2ECV_torus_24576, 24, 12, 36, 65, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_unstructured_torus_cpu_kfirst_runtimes.begin(),
        nabla4_benchmark_unstructured_torus_cpu_kfirst_runtimes.end());
    std::cout << "Unstructured Torus 24576 cpu_kfirst nabla4 median runtime: "
              << nabla4_benchmark_unstructured_torus_cpu_kfirst_runtimes
                     [nabla4_benchmark_unstructured_torus_cpu_kfirst_runtimes.size() / 2]
              << std::endl;

    auto nabla4_benchmark_structured_torus_naive_runtimes =
        nabla4_benchmark_structured_torus_naive(24, 12, 36, 65, 4, 3, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_structured_torus_naive_runtimes.begin(),
        nabla4_benchmark_structured_torus_naive_runtimes.end());
    std::cout
        << "Structured Torus 24576 naive nabla4 median runtime: "
        << nabla4_benchmark_structured_torus_naive_runtimes[nabla4_benchmark_structured_torus_naive_runtimes.size() / 2]
        << std::endl;
    auto nabla4_benchmark_structured_torus_cpu_ifirst_runtimes =
        nabla4_benchmark_structured_torus_cpu_ifirst(24, 12, 36, 65, 4, 3, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_structured_torus_cpu_ifirst_runtimes.begin(),
        nabla4_benchmark_structured_torus_cpu_ifirst_runtimes.end());
    std::cout << "Structured Torus 24576 cpu_ifirst nabla4 median runtime: "
              << nabla4_benchmark_structured_torus_cpu_ifirst_runtimes
                     [nabla4_benchmark_structured_torus_cpu_ifirst_runtimes.size() / 2]
              << std::endl;
    auto nabla4_benchmark_structured_torus_cpu_kfirst_runtimes =
        nabla4_benchmark_structured_torus_cpu_kfirst(24, 12, 36, 65, 4, 3, 4, repetitions, dry_runs);
    std::sort(nabla4_benchmark_structured_torus_cpu_kfirst_runtimes.begin(),
        nabla4_benchmark_structured_torus_cpu_kfirst_runtimes.end());
    std::cout << "Structured Torus 24576 cpu_kfirst nabla4 median runtime: "
              << nabla4_benchmark_structured_torus_cpu_kfirst_runtimes
                     [nabla4_benchmark_structured_torus_cpu_kfirst_runtimes.size() / 2]
              << std::endl;
}
