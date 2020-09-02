/*
 *  Copyright (c), 2020, EPFL - Blue Brain Project
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#pragma once

#include <Eigen/Eigen>

namespace HighFive {

namespace details {

template<typename T, int M, int N>
struct data_converter<Eigen::Matrix<T, M, N>> {
    using value_type = Eigen::Matrix<T, M, N>;
    using dataspace_type = T;
    using h5_type = T;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
    : _space(space)
    , _dims(dims)
    {
        if (_dims.size() > 2) { // Can be vector or matrix
            throw std::string("Invalid number of dimensions for eigen matrix");
        }
    }

    void allocate(value_type& val) {
        val.resize(static_cast<typename value_type::Index>(_dims[0]),
                   static_cast<typename value_type::Index>(_dims.size() > 1 ? _dims[1] : 1));
    }

    static std::vector<size_t> get_size(const value_type& val) {
        return std::vector<size_t>{static_cast<size_t>(val.rows()), static_cast<size_t>(val.cols())};
    }

    static dataspace_type* get_pointer(value_type& val) {
        return val.data();
    }

    static const dataspace_type* get_pointer(const value_type& val) {
        return val.data();
    }

    inline void unserialize(value_type& vec, const dataspace_type* data) {
        for (unsigned int i = 0; i < vec.rows(); ++i) {
            for (unsigned int j = 0; j < vec.cols(); ++j) {
                vec(i, j) = data[i * vec.cols() + j];
            }
        }
    }

    inline void serialize(const value_type& vec, dataspace_type* data) const {
        for (unsigned int i = 0; i < vec.rows(); ++i) {
            for (unsigned int j = 0; j < vec.cols(); ++j) {
                data[i * vec.cols() + j] = vec(i, j);
            }
        }
    }

    const DataSpace& _space;
    std::vector<size_t> _dims;
    size_t _number_of_element = 2;

    static constexpr size_t number_of_dims = 2;
};

template <typename S, int M, int N>
struct h5_continuous<Eigen::Matrix<S, M, N>> :
    std::true_type {};

}  // namespace details

}  // namespace HighFive
