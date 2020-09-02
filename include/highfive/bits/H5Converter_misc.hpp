/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */

/*  INTERFACES
struct data_converter {
  using value_type;
  using dataspace_type;

  // Size for HDF5
  static std::vector<size_t> get_size(const value_type&);
  // Number of elements for C++
  static size_t _number_of_element;

  // Before reading for creating a type
  void allocate(value_type&) const;

  // Only for continuous
  // Reading
  static dataspace_type* get_pointer(value_type&);
  // Writing
  static const dataspace_type* get_pointer(const value_type&);

  // Before writing non-continuous
  inline void serialize(const value_type&, dataspace_type*);

  // After reading non-continuous
  inline void unserialize(value_type&, const dataspace_type*);

  // Use to know how to convert things
  DataType h5_type;

  static constexpr size_t number_of_dims;
};
*/

#ifndef H5CONVERTER_MISC_HPP
#define H5CONVERTER_MISC_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>

#ifdef H5_USE_BOOST
// starting Boost 1.64, serialization header must come before ublas
#include <boost/serialization/vector.hpp>
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#endif

#include <H5Dpublic.h>
#include <H5Ppublic.h>

#include "../H5Reference.hpp"
#include "../H5DataType.hpp"
#include "H5Utils.hpp"

namespace HighFive {

namespace details {
template <typename T>
struct h5_pod :
    std::is_pod<T> {};

template <>
struct h5_pod<bool> :
    std::false_type {};

// contiguous pair of T
// template <typename T>
// struct h5_pod<std::complex<T>> :
//     std::true_type {};

template <typename T>
struct h5_continuous :
    std::false_type {};

template <typename S>
struct h5_continuous<std::vector<S>> :
    std::integral_constant<bool, h5_pod<S>::value> {};

template <typename T, size_t N>
struct h5_continuous<T[N]> :
    std::integral_constant<bool, h5_pod<T>::value> {};

template <typename S, size_t N>
struct h5_continuous<std::array<S, N>> :
    std::integral_constant<bool, h5_pod<S>::value> {};

template <typename T, size_t Dims>
struct h5_continuous<boost::multi_array<T, Dims>> :
    std::integral_constant<bool, h5_pod<T>::value> {};

template <typename T>
struct h5_continuous<boost::numeric::ublas::matrix<T>> :
    std::integral_constant<bool, h5_pod<T>::value> {};

// template <typename T, int M, int N>
// struct h5_continuous<Eigen::Matrix<T, M, N>> :
//     std::true_type {};

template <typename T>
struct h5_non_continuous :
    std::integral_constant< bool, !h5_continuous<T>::value> {};

size_t get_number_of_elements(const std::vector<size_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), size_t{1u}, std::multiplies<size_t>());
}

template <typename T>
struct data_converter {
    using value_type = T;
    using dataspace_type = T;
    using h5_type = value_type;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
      : _space(space)
    {
        if (!dims.empty()) {
            throw std::string("Invalid number of elements for scalar");
        }
    }

    void allocate(value_type&) const {
        // pass
    }

    static std::vector<size_t> get_size(const value_type& val) {
        return std::vector<size_t>{};
    }

    static dataspace_type* get_pointer(value_type& val) {
        return &val;
    }

    static const dataspace_type* get_pointer(const value_type& val) {
        return &val;
    }

    inline void unserialize(value_type& scalar, const dataspace_type* data) {
        scalar = *data;
    }

    inline void serialize(const value_type& scalar, dataspace_type* data) {
        *data = scalar;
    }

    const DataSpace& _space;
    size_t _number_of_element = 1;

    static constexpr size_t number_of_dims = 0;
};

template <typename T>
struct data_converter<std::complex<T>> {
    using value_type = std::complex<T>;
    using dataspace_type = std::complex<T>;
    using h5_type = value_type;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
      : _space(space)
    {
        if (!dims.empty()) {
            throw std::string("Invalid number of elements for complex");
        }
    }

    void allocate(value_type&) {
        // pass
    }

    static std::vector<size_t> get_size(const value_type& val) {
        return std::vector<size_t>{};
    }

    static dataspace_type* get_pointer(value_type& val) {
        return &val;
    }

    static const dataspace_type* get_pointer(const value_type& val) {
        return &val;
    }

    inline void unserialize(value_type& scalar, const dataspace_type* data) {
        scalar = *data;
    }

    inline void serialize(const value_type& scalar, dataspace_type* data) {
        *data = scalar;
    }

    const DataSpace& _space;
    size_t _number_of_element = 1;

    static constexpr size_t number_of_dims = 0;
};

template <>
struct data_converter<Reference> {
    using value_type = Reference;
    using dataspace_type = hobj_ref_t;
    using h5_type = value_type;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
      : _space(space)
    {
        if (!dims.empty()) {
            throw std::string("Invalid number of elements for Reference");
        }
    }

    void allocate(value_type&) {
        // pass
    }

    static std::vector<size_t> get_size(const value_type& val) {
        return std::vector<size_t>{};
    }

    static dataspace_type* get_pointer(value_type& val) {
        return nullptr;
    }

    static const dataspace_type* get_pointer(const value_type& val) {
        return nullptr;
    }

    inline void unserialize(value_type& scalar, const dataspace_type* data) {
        scalar = Reference(*data);
    }

    inline void serialize(const value_type& scalar, dataspace_type* data) {
        scalar.create_ref(data);
    }

    const DataSpace& _space;
    size_t _number_of_element = 1;

    static constexpr size_t number_of_dims = 0;
};

template<>
struct data_converter<std::string> {
    using value_type = std::string;
    using dataspace_type = const char*;
    using h5_type = value_type;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
      : _space(space)
    {
        if (!dims.empty()) {
            throw std::string("Invalid number of elements for variable string");
        }
    }

    void allocate(value_type&) {
        // pass
    }

    static std::vector<size_t> get_size(const value_type& val) {
        return std::vector<size_t>{};
    }

    static dataspace_type* get_pointer(value_type& val) {
        // .data() return a const pointer until c++17
        throw std::string("Invalid get_pointer on std::string");
    }

    static const dataspace_type* get_pointer(const value_type& val) {
        static dataspace_type _c_str = const_cast<char*>(val.c_str());
        return &_c_str;
    }

    inline void unserialize(value_type& scalar, const dataspace_type* data) {
        scalar = std::string(*data);
    }

    inline void serialize(const value_type& scalar, dataspace_type* data) {
        dataspace_type _c_str = const_cast<char*>(scalar.c_str());
        *data = _c_str;
    }

    const DataSpace& _space;
    size_t _number_of_element = 1;

    static constexpr size_t number_of_dims = 0;
};

template <typename T, typename U, size_t size>
struct data_converter_container_base {
    using value_type = T;
    using inner_type = data_converter<U>;
    using dataspace_type = typename inner_type::dataspace_type;
    using h5_type = typename inner_type::h5_type;

    inline data_converter_container_base(const DataSpace& space, const std::vector<size_t>& dims)
      : _space(space)
      , _dims(dims)
      , _inner_converter(space, std::vector<size_t>(dims.begin() + size, dims.end()))
      , _number_of_element(_dims[0] * _inner_converter._number_of_element)
    {}

    virtual ~data_converter_container_base() = default;
    virtual void resize(value_type& val) = 0;
    virtual size_t get_total_size(const value_type& val) = 0;
    virtual U* get_elem(value_type& val) = 0;
    virtual const U* get_elem(const value_type& val) = 0;
    
    void allocate(value_type& val) {
        resize(val);
        for(size_t i = 0; i < get_total_size(val); ++i) {
            _inner_converter.allocate(get_elem(val)[i]);
        }
    }

    // Internal function that take data and put it in vec
    inline void unserialize(value_type& vec, const dataspace_type* data) {
        for (unsigned int i = 0; i < get_total_size(vec); ++i) {
            _inner_converter.unserialize(get_elem(vec)[i], data + _inner_converter._number_of_element * i); 
        }
    }

    inline void serialize(const value_type& scalar, dataspace_type* data) {
         for (size_t i = 0; i < get_total_size(scalar); ++i) {
             _inner_converter.serialize(get_elem(scalar)[i], data + _inner_converter._number_of_element * i);
         }
    }

    const DataSpace& _space;
    std::vector<size_t> _dims;
    inner_type _inner_converter;
    size_t _number_of_element;

    static constexpr size_t number_of_dims = size + inner_type::number_of_dims;
};

template <typename T>
struct data_converter<std::vector<T>>: public data_converter_container_base<std::vector<T>, T, 1> {
    using parent = data_converter_container_base<std::vector<T>, T, 1>;
    using typename parent::value_type;
    using typename parent::inner_type;
    using typename parent::dataspace_type;
    using typename parent::h5_type;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
      : parent(space, dims)
    {}

    void resize(value_type& val) override {
        val.resize(parent::_dims[0]);
    }

    size_t get_total_size(const value_type& val) override {
        return val.size();
    }

    T* get_elem(value_type& val) override {
        return val.data();
    }

    const T* get_elem(const value_type& val) override {
        return val.data();
    }

    static T* get_pointer(value_type& val) {
        return val.data();
    }

    static const T* get_pointer(const value_type& val) {
        return val.data();
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(get_pointer(val)[0]);
        ret.insert(ret.begin(), val.size());
        return ret;
    }
};

template <typename T, size_t N>
struct data_converter<std::array<T, N>>: public data_converter_container_base<std::array<T, N>, T, 1> {
    using parent = data_converter_container_base<std::array<T, N>, T, 1>;
    using typename parent::value_type;
    using typename parent::inner_type;
    using typename parent::dataspace_type;
    using typename parent::h5_type;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
      : parent(space, dims)
    {
        if (dims[0] != N) {
            throw std::runtime_error("Size of the std::array is not valid");
        }
    }

    void resize(value_type& val) override {
        // statically allocated
    }

    size_t get_total_size(const value_type& val) override {
        return val.size();
    }

    T* get_elem(value_type& val) override {
        return val.data();
    }

    const T* get_elem(const value_type& val) override {
        return val.data();
    }

    static T* get_pointer(value_type& val) {
        return val.data();
    }

    static const T* get_pointer(const value_type& val) {
        return val.data();
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(get_pointer(val)[0]);
        ret.insert(ret.begin(), val.size());
        return ret;
    }
};

#ifdef H5_USE_BOOST
// apply conversion to boost multi array
template <typename T, std::size_t Dims>
struct data_converter<boost::multi_array<T, Dims>>: public data_converter_container_base<boost::multi_array<T, Dims>, T, Dims> {
    using parent = data_converter_container_base<boost::multi_array<T, Dims>, T, Dims>;
    using typename parent::value_type;
    using typename parent::inner_type;
    using typename parent::dataspace_type;
    using typename parent::h5_type;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
      : parent(space, dims)
    {
        if (dims.size() < Dims) {
            throw std::string("Invalid number of dimensions");
        }
    }

    void resize(value_type& val) override {
        boost::array<typename value_type::index, Dims> ext;
        std::copy(parent::_dims.begin(), parent::_dims.begin() + Dims, ext.begin());
        val.resize(ext);
    }

    size_t get_total_size(const value_type& val) override {
        return val.num_elements();
    }

    T* get_elem(value_type& val) override {
        return val.data();
    }

    const T* get_elem(const value_type& val) override {
        return val.data();
    }

    static T* get_pointer(value_type& val) {
        return val.data();
    }

    static const T* get_pointer(const value_type& val) {
        return val.data();
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(get_pointer(val)[0]);
        for (unsigned i = Dims; i > 0; --i) {
            ret.insert(ret.begin(), val.shape()[i-1]);
        }
        return ret;
    }
};

template <typename T>
struct data_converter<boost::numeric::ublas::matrix<T>>: data_converter_container_base<boost::numeric::ublas::matrix<T>, T, 2> {
    using parent = data_converter_container_base<boost::numeric::ublas::matrix<T>, T, 2>;
    using typename parent::value_type;
    using typename parent::inner_type;
    using typename parent::dataspace_type;
    using typename parent::h5_type;

    inline data_converter(const DataSpace& space, const std::vector<size_t>& dims)
      : parent(space, dims)
    {
        if (dims.size() < 2) {
            throw std::string("Invalid number of dimensions");
        }
    }

    void resize(value_type& val) override {
        val.resize(parent::_dims[0], parent::_dims[1], false);
        val(0, 0) = 0;
    }

    size_t get_total_size(const value_type& val) override {
        return val.size1() * val.size2();
    }

    T* get_elem(value_type& val) override {
        return &val(0, 0);
    }

    const T* get_elem(const value_type& val) override {
        return &val(0, 0);
    }

    static T* get_pointer(value_type& val) {
        return &val(0, 0);
    }

    static const T* get_pointer(const value_type& val) {
        return &val(0, 0);
    }

    static std::vector<size_t> get_size(const value_type& val) {
        auto ret = inner_type::get_size(get_pointer(val)[0]);
        ret.insert(ret.begin(), val.size2());
        ret.insert(ret.begin(), val.size1());
        return ret;
    }
};
#endif
}  // namespace details

std::vector<size_t> real_dims(const std::vector<size_t>& from, size_t to_size) {
    if (from.size() <= to_size) { // We can do nothing
        return from;
    }

    auto distance = static_cast<long int>(from.size() - to_size);
    bool correct = true;
    for (long int i = 0; i < distance; ++i) {
        if (from[static_cast<size_t>(i)] != 1) {
            correct = false;
            break;
        }
    }
    if (correct) {
        return std::vector<size_t>(from.begin() + distance, from.end());
    }

    correct = true;
    for (long int i = distance; i > 0; --i) {
        if (from[static_cast<size_t>(i)] != 1) {
            correct = false;
            break;
        }
    }
    if (correct) {
        return std::vector<size_t>(from.begin(), from.end() - distance);
    }
    return from;
}

template <typename T, typename Enable = void>
class TransformRead;

template <typename T>
class TransformRead<T, typename std::enable_if<details::h5_non_continuous<T>::value>::type> {
  public:
    using dataspace_type = typename details::data_converter<T>::dataspace_type;
    using Conv = details::data_converter<T>;
    using h5_type = typename Conv::h5_type;

    TransformRead(const DataSpace& space)
        : _space(space)
        , _converter(_space, real_dims(space.getDimensions(), Conv::number_of_dims))
    {
#ifdef H5_ENABLE_ERROR_ON_COPY
#error You are using a non continuous data type and so data will be copied
#endif
    }

    dataspace_type* get_pointer() {
        _vec.resize(details::get_number_of_elements(_space.getDimensions()));
        return _vec.data();
    }

    const dataspace_type* get_pointer() const {
        _vec.resize(details::get_number_of_elements(_space.getDimensions()));
        return _vec.data();
    }

    T transform_read() {
        T _data;
        _converter.allocate(_data);
        _converter.unserialize(_data, _vec.data());
        return _data;
    }

  private:
    // Continuous vector for data
    std::vector<dataspace_type> _vec;

    const DataSpace& _space;
    Conv _converter;
  public:
    static constexpr size_t number_of_dims = Conv::number_of_dims;
    DataType _h5_type = create_and_check_datatype<h5_type>();
};

template<typename T>
class TransformRead<T, typename std::enable_if<details::h5_continuous<T>::value>::type> {
  public:
    using dataspace_type = typename details::data_converter<T>::dataspace_type;
    using Conv = details::data_converter<T>;
    using h5_type = typename Conv::h5_type;

    TransformRead(const DataSpace& space)
        : _space(space)
        , _converter(_space, real_dims(space.getDimensions(), Conv::number_of_dims))
    {
    }

    dataspace_type* get_pointer() {
        _converter.allocate(_data);
        return _converter.get_pointer(_data);
    }

    T transform_read() {
        return _data;
    }

  private:
    T _data;
    const DataSpace& _space;
    Conv _converter;
  public:
    static constexpr size_t number_of_dims = Conv::number_of_dims;
    DataType _h5_type = create_and_check_datatype<h5_type>();
};

template <typename T, typename Enable = void>
class TransformWrite;

template <typename T>
class TransformWrite<T, typename std::enable_if<details::h5_non_continuous<T>::value>::type> {
  public:
    using dataspace_type = typename details::data_converter<T>::dataspace_type;
    using Conv = details::data_converter<T>;
    using h5_type = typename Conv::h5_type;

    TransformWrite(const DataSpace& space, const T& value)
        : _dims(real_dims(space.getDimensions(), Conv::number_of_dims))
        , _converter(space, _dims)
    {
#ifdef H5_ENABLE_ERROR_ON_COPY
#error You are using a non continuous data type and so data will be copied
#endif
        _vec.resize(details::get_number_of_elements(_dims));
        _converter.serialize(value, _vec.data());
    }

    const dataspace_type* get_pointer() const {
        return _vec.data();
    }

  private:
    std::vector<size_t> _dims;
    Conv _converter;
    std::vector<dataspace_type> _vec;
  public:
    static constexpr size_t number_of_dims = Conv::number_of_dims;
    DataType _h5_type = create_and_check_datatype<h5_type>();
};

template <typename T>
class TransformWrite<T, typename std::enable_if<details::h5_continuous<T>::value>::type> {
  public:
    using dataspace_type = typename details::data_converter<T>::dataspace_type;
    using Conv = details::data_converter<T>;
    using h5_type = typename Conv::h5_type;

    TransformWrite(const DataSpace& space, const T& value)
        : _dims(real_dims(space.getDimensions(), Conv::number_of_dims))
        , _converter(space, _dims)
        , _data(value)
    {}

    const dataspace_type* get_pointer() {
        return _converter.get_pointer(_data);
    }

  private:
    std::vector<size_t> _dims;
    Conv _converter;
    const T& _data;
  public:
    static constexpr size_t number_of_dims = Conv::number_of_dims;
    DataType _h5_type = create_and_check_datatype<h5_type>();
};

// Wrappers to have template deduction, that are not available with class before C++17
template <typename T>
TransformWrite<T> make_transform_write(const DataSpace& space, const T& value) {
    return TransformWrite<T>{space, value};
}

template <typename T>
TransformRead<T> make_transform_read(const DataSpace& space) {
    return TransformRead<T>{space};
}

template <typename T>
std::vector<size_t> compute_dims(const T& value) {
    return details::data_converter<T>::get_size(value);
}

}  // namespace HighFive

#ifdef H5_USE_EIGEN
#include "H5ConverterEigen_misc.hpp"
#endif

#endif // H5CONVERTER_MISC_HPP
