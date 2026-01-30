#include "distance_calculator.hpp"

Eigen::MatrixXf get_dist_matrix_no_box(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2) {
    // column-major
    Eigen::MatrixXf coords1_expanded_row = coord1.replicate(coord2.cols(), 1);
    coords1_expanded_row.resize(ncoords, coord1.cols() * coord2.cols());
    Eigen::MatrixXf coords2_expanded_col = coord2.replicate(1, coord1.cols());
    
    Eigen::MatrixXf diff = coords2_expanded_col - coords1_expanded_row;
    Eigen::MatrixXf dist_matrix = diff.colwise().norm().reshaped(coord1.cols(), coord2.cols());

    return dist_matrix;
}

std::pair<Eigen::MatrixXf, Eigen::Vector<char, ncoords> > 
    get_dist_matrix_orthorhombic(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2, const Eigen::Matrix3f box) {
    // column-major
    static Eigen::MatrixXf dist_matrix;
    static Eigen::MatrixXf coords1_expanded_row;
    static Eigen::MatrixXf coords2_expanded_col;
    static Eigen::MatrixXf diff;
    static Eigen::MatrixXf shift_real;
    Eigen::Vector3f diag = box.diagonal();
    coords1_expanded_row = coord1.replicate(coord2.cols(), 1);
    coords1_expanded_row.resize(ncoords, coord1.cols() * coord2.cols());
    coords2_expanded_col = coord2.replicate(1, coord1.cols());
    diff = coords2_expanded_col - coords1_expanded_row;
    shift_real = - (diff.array().colwise() / diag.array()).round();
    diff.array() += shift_real.array().colwise() * diag.array();
    dist_matrix = diff.colwise().norm(); // Eigen::RowVector, shape (1, coord1.cols() * coord2.cols())
    Eigen::MatrixXf::Index min_dist_pos;
    dist_matrix.row(0).minCoeff(& min_dist_pos);
    dist_matrix.resize(coord1.cols(), coord2.cols());
    Eigen::Vector<char, ncoords> shift_min = shift_real.col(min_dist_pos).cast<char>();
    return std::make_pair(dist_matrix, shift_min);
}

std::pair<Eigen::MatrixXf, Eigen::Vector<char, ncoords> > 
    get_dist_squared_matrix_orthorhombic(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2, const Eigen::Matrix3f box) {
    // column-major
    static Eigen::MatrixXf squared_dist_matrix;
    static Eigen::MatrixXf coords1_expanded_row;
    static Eigen::MatrixXf coords2_expanded_col;
    static Eigen::MatrixXf diff;
    static Eigen::MatrixXf shift_real;
    Eigen::Vector3f diag = box.diagonal();
    coords1_expanded_row = coord1.replicate(coord2.cols(), 1);
    coords1_expanded_row.resize(ncoords, coord1.cols() * coord2.cols());
    coords2_expanded_col = coord2.replicate(1, coord1.cols());
    diff = coords2_expanded_col - coords1_expanded_row;
    shift_real = - (diff.array().colwise() / diag.array()).round();
    diff.array() += shift_real.array().colwise() * diag.array();
    squared_dist_matrix = diff.colwise().squaredNorm(); // Eigen::RowVector, shape (1, coord1.cols() * coord2.cols())
    Eigen::MatrixXf::Index min_dist_pos;
    squared_dist_matrix.row(0).minCoeff(& min_dist_pos);
    squared_dist_matrix.resize(coord1.cols(), coord2.cols());
    Eigen::Vector<char, ncoords> shift_min = shift_real.col(min_dist_pos).cast<char>();
    return std::make_pair(squared_dist_matrix, shift_min);
}
