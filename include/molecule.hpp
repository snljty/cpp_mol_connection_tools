#pragma once
#ifndef __MOLECULE_HPP__
#define __MOLECULE_HPP__

#include <string>
#include <vector>

#include <Eigen/Dense>

class Molecule {
public:
    int natoms;
    std::vector<std::string> elements;
    Eigen::MatrixXf coordinates;

    void resize(int natoms);
    void write_gjf(const std::string &ofilename);
};

#endif // __MOLECULE_HPP__

