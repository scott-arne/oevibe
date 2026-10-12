#include "engines/ElectronicProperties.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace OEVIBE {
namespace detail {

int homo_index(const std::vector<double>& spin_summed_occupations) {
    for (std::size_t k = spin_summed_occupations.size(); k-- > 0;) {
        if (spin_summed_occupations[k] > HOMO_OCCUPATION_THRESHOLD) {
            return static_cast<int>(k);
        }
    }
    return -1;
}

std::vector<BondOrder> wiberg_pairs(const std::vector<double>& matrix, unsigned num_atoms) {
    const std::size_t n = num_atoms;
    if (matrix.size() != n * n) {
        throw std::invalid_argument("a bond-order matrix for " + std::to_string(n) + " atoms has " +
                                    std::to_string(n * n) + " values, not " +
                                    std::to_string(matrix.size()));
    }
    std::vector<BondOrder> pairs;
    for (unsigned i = 0; i < num_atoms; ++i) {
        for (unsigned j = i + 1; j < num_atoms; ++j) {
            const double order = matrix[i * n + j];
            if (!std::isfinite(order) || order >= WIBERG_REPORT_THRESHOLD) {
                pairs.push_back({i, j, order});
            }
        }
    }
    return pairs;
}

}  // namespace detail
}  // namespace OEVIBE
