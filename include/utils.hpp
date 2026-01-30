#pragma once
#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <vector>
#include <array>
#include <string>

#include <Eigen/Dense>

using coord_direction = enum {coord_x, coord_y, coord_z};

constexpr int ncoords = 3;

constexpr size_t max_element_num = 1 + 118;

extern const std::vector<std::string> elements_names;

extern const std::array<float, max_element_num> elements_van_der_Waals_radius;

extern const std::array<float, max_element_num> elements_covalence_radius;

void getline_check(std::ifstream &ifile, std::string &line);

std::vector<int> indices_str_to_list_from_0(const std::string &indices);

void indices_str_to_list_from_0(const std::string &indices, std::vector<int> &result);

std::string list_to_indices_str_from_1(const std::vector<int> &nums);

void list_to_indices_str_from_1(const std::vector<int> &nums, std::string &result);

bool is_box_orthorhombic(const Eigen::Matrix3f &box);

void print_box(const Eigen::Matrix3f &box);

#endif // __UTILS_HPP__

