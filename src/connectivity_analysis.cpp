#include "distance_calculator.hpp"
#include "molecules_box.hpp"

#ifndef NDEBUG
#include <fmt/format.h>
#endif

void MoleculesBox::get_intramolecule_connectivity(int imol, float scaler_tol) {
    // get distance matrix, no PBC considered
    static Eigen::MatrixXf dist_matrix_intra;
    dist_matrix_intra = get_dist_matrix_no_box(coordinates[imol], coordinates[imol]);

    intramolecule_connectivity = (dist_matrix_intra.array() <= (atomic_covalence_radius_sum * scaler_tol).array()).cast<unsigned char>();
    intramolecule_connectivity.diagonal().setZero();
}

void MoleculesBox::get_intermolecule_connectivity_of_selected_part() {
    static std::pair<Eigen::MatrixXf, Eigen::Vector<char, ncoords> > current;
    static Eigen::MatrixXi short_contact;

    for (int jmol = 0; jmol < nmols; ++ jmol) {
        for (int imol = jmol + 1; imol < nmols; ++ imol) {
            #ifndef NDEBUG
            fmt::print("\rhandling imol = {:3d}, jmol = {:3d}", imol, jmol);
            std::fflush(stdout);
            #endif
            current = get_dist_squared_matrix_orthorhombic(coordinates_selected[imol], coordinates_selected[jmol], box);
            short_contact = (current.first.array() < atomic_van_der_Waals_radius_selected_sum_squared.array()).cast<int>();
            num_intermolecule_short_contact_atoms(imol, jmol) = 
                short_contact.rowwise().any().sum() + 
                short_contact.colwise().any().sum();
            num_intermolecule_short_contact_pairs(imol, jmol) = short_contact.sum();
            shifts_min[coord_x](imol, jmol) = current.second[coord_x];
            shifts_min[coord_y](imol, jmol) = current.second[coord_y];
            shifts_min[coord_z](imol, jmol) = current.second[coord_z];
        }
    }
    #ifndef NDEBUG
    fmt::print("\r{:100s}\r", "");
    std::fflush(stdout);
    #endif

    for (int jmol = 0; jmol < nmols; ++ jmol) {
        for (int imol = jmol + 1; imol < nmols; ++ imol) {
            num_intermolecule_short_contact_atoms(jmol, imol) = num_intermolecule_short_contact_atoms(imol, jmol);
            num_intermolecule_short_contact_pairs(jmol, imol) = num_intermolecule_short_contact_pairs(imol, jmol);
            shifts_min[coord_x](jmol, imol) = shifts_min[coord_x](imol, jmol);
            shifts_min[coord_y](jmol, imol) = shifts_min[coord_y](imol, jmol);
            shifts_min[coord_z](jmol, imol) = shifts_min[coord_z](imol, jmol);
        }
    }
}

void MoleculesBox::get_a_nice_molecule() {
    index_of_a_nice_molecule = -1;
    std::vector<unsigned char> visited(natoms_per_mol);
    bool all_connected;
    for (int imol = 0; imol < nmols; ++ imol) {
        std::fill(visited.begin(), visited.end(), 0);
        get_intramolecule_connectivity(imol);
        connection_search(visited, 0);
        all_connected = true;
        for (int i = 0; i < natoms_per_mol; ++ i) {
            if (! visited[i]) all_connected = false;
        }
        if (all_connected) {
            index_of_a_nice_molecule = imol;
            break;
        }
    }
    if (index_of_a_nice_molecule < 0) {
        throw std::runtime_error("Cannot find even one whole molecule.");
    }
}

void MoleculesBox::connection_search(std::vector<unsigned char> &visited, int i) {
    visited[i] = 1;
    for (int j = 0; j < natoms_per_mol; ++ j) {
        if (! visited[j] && intramolecule_connectivity(i, j)) {
            connection_search(visited, j);
        }
    }
}
