#include <vector>

#include "wrapper.hpp"

int main(void) {
    std::vector<std::vector<double>> E2C2V{{0, 1, 4, 6},
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
    const auto nabla4_benchmark_runtimes = nabla4_benchmark(E2C2V, 18, 9, 18, 65, 4, 5);
    for (const auto& runtime : nabla4_benchmark_runtimes) {
        std::cout << runtime << std::endl;
    }
}
