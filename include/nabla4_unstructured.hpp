#include <vector>
#include <iostream>

#include <thread>

class nabla4_unstructured
{
private:
    std::vector<std::vector<double>> e2c2v;
    std::size_t CellDim;
    std::size_t EdgeDim;
    std::size_t VertexDim;
    std::size_t KDim;
    std::size_t ECVDim;
public:
    nabla4_unstructured(std::vector<std::vector<double>>& e2c2v, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim) : e2c2v(e2c2v), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim) {};
    void run() {
        using namespace std::chrono_literals;
        std::cout << "Running nabla4_unstructured benchmark" << std::endl;
        std::this_thread::sleep_for(2ms);
    };
};
