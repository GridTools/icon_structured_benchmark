#include <vector>
#include <random>

class RandomUniformUtils {

    std::random_device _rd{};
    std::mt19937 _gen{_rd()};
    std::uniform_real_distribution<double> _dist;

    template<typename T>
    double get() {
        return static_cast<T>(_dist(_gen));
    }

public:

    RandomUniformUtils(double low = 0.0, double high = 10.0) {
        std::uniform_real_distribution<double>::param_type param(low, high);
        _dist.param(param);
    }

    template<typename T = double>
    std::vector<T> random_init_vec_1d(std::size_t domain_size) {
        std::vector<T> vec;
        vec.reserve(domain_size);
        for (std::size_t i; i < domain_size; ++i) {
            vec[i] = get<T>();
        }
        return vec;
    }

    template<typename T = double>
    std::vector<std::vector<T>> random_init_vec_2d(std::size_t domain_i_size, std::size_t domain_j_size) {
        std::vector<std::vector<T>> vec;
        vec.reserve(domain_i_size);
        for (auto& jvec : vec) {
            jvec.reserve(domain_j_size);
        }
        for (std::size_t i; i < domain_i_size; ++i) {
            for (std::size_t j; j < domain_j_size; ++j) {
                vec[i][j] = get<T>();
            }
        }
        return vec;
    }

};
