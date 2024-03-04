#include <vector>
#include <iostream>
#include <thread>

#include "random_init.hpp"

class nabla4_unstructured
{
private:
    std::vector<std::vector<double>> e2c2v;
    std::size_t CellDim;
    std::size_t EdgeDim;
    std::size_t VertexDim;
    std::size_t KDim;
    std::size_t ECVDim;
    std::vector<std::vector<float>> u_vert;
    std::vector<std::vector<float>> v_vert;
    std::vector<double> primal_normal_vert_v1;
    std::vector<double> primal_normal_vert_v2;
    std::vector<std::vector<double>> z_nabla2_e;
    std::vector<double> inv_vert_vert_length;
    std::vector<double> inv_primal_edge_length;

    /// Random number utilities
    RandomUniformUtils rand_utils{-2.0, 2.0};

    /// Initialize vectors needed to execute kernel with random numbers
    void init() {
        u_vert = rand_utils.random_init_vec_2d<float>(VertexDim, KDim);
        v_vert = rand_utils.random_init_vec_2d<float>(VertexDim, KDim);
        primal_normal_vert_v1 = rand_utils.random_init_vec_1d(ECVDim);
        primal_normal_vert_v2 = rand_utils.random_init_vec_1d(ECVDim);
        z_nabla2_e = rand_utils.random_init_vec_2d(EdgeDim, KDim);
        inv_vert_vert_length = rand_utils.random_init_vec_1d(EdgeDim);
        inv_primal_edge_length = rand_utils.random_init_vec_1d(EdgeDim);
    }

public:

    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_unstructured(std::vector<std::vector<double>>& e2c2v, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim) : e2c2v(e2c2v), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim) {
        init();
    };

    /// Compute function timed for benchmarking
    void run() {
        using namespace std::chrono_literals;
        std::cout << "Running nabla4_unstructured benchmark" << std::endl;
        std::this_thread::sleep_for(2ms);
    };
};
