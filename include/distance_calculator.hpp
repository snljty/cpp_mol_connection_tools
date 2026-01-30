#pragma once
#ifndef __DISTANCE_CALCULATOR_HPP
#define __DISTANCE_CALCULATOR_HPP

#include "utils.hpp"

Eigen::MatrixXf get_dist_matrix_no_box(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2);

std::pair<Eigen::MatrixXf, Eigen::Vector<char, ncoords> > 
    get_dist_matrix_orthorhombic(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2, const Eigen::Matrix3f box);
    // return dist_matrix and shift_min

std::pair<Eigen::MatrixXf, Eigen::Vector<char, ncoords> > 
    get_dist_squared_matrix_orthorhombic(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2, const Eigen::Matrix3f box);
    // return dist_squared_matrix and shift_min

#endif // __DISTANCE_CALCULATOR_HPP

