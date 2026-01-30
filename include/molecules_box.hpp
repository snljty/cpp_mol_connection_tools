#pragma once
#ifndef __MOLECULES_BOX_HPP__
#define __MOLECULES_BOX_HPP__

#include "element_table.hpp"
#include "molecule.hpp"
#include "utils.hpp"

#include <vector>
#include <string>

#include <Eigen/Dense>

class MoleculesBox {
public:
    Element_table element_table;
    int natoms;
    int nmols;
    int natoms_per_mol;
    int natoms_selected = 0;
    double timestamp = 0.;
    std::vector<std::string> elements;
    std::vector<int> atomic_indices;
    Eigen::VectorXf atomic_van_der_Waals_radius;
    Eigen::VectorXf atomic_covalence_radius;
    Eigen::MatrixXf atomic_covalence_radius_sum;
    std::vector<Eigen::MatrixXf> coordinates;
    std::vector<int> selection;
    std::vector<int> atomic_indices_selected;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> intramolecule_connectivity;
    Eigen::VectorXf atomic_van_der_Waals_radius_selected;
    Eigen::MatrixXf atomic_van_der_Waals_radius_selected_sum;
    Eigen::MatrixXf atomic_van_der_Waals_radius_selected_sum_squared;
    std::vector<Eigen::MatrixXf> coordinates_selected;
    Eigen::Matrix3d box_d;
    Eigen::Matrix3f box;
    Eigen::MatrixXi num_intermolecule_short_contact_atoms;
    Eigen::MatrixXi num_intermolecule_short_contact_pairs;
    std::array<Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic>, ncoords> shifts_min;
    std::vector<int> backbone;
    std::vector<int> backbone_noH;
    Molecule monomer;
    Molecule monomer_methyl;
    Molecule monomer_backbone_no_hydrogen;
    Molecule dimer;
    Molecule dimer_methyl;
    Molecule dimer_backbone_no_hydrogen;
    std::vector<int> alkyl_connection_site;
    std::vector<std::vector<int> > alkyl_connection_site_connected_alkyl;
    int natoms_per_trimmed_mol;
    int natoms_truncated_methyl_total;
    int index_of_a_nice_molecule = -1;

    // MoleculesBox() = default;
    MoleculesBox(const std::string &ifilename, int nmols=0);
    // ~MoleculesBox() = default;
    void read_gro(const std::string &ifilename, int nmols=0);
    void set_selection(const std::string &selection_str);
    void set_selection(const std::vector<int> &selection_list);
    void set_selection_internal();
    void get_intramolecule_connectivity(int imol=0, float scaler_tol=1.15f);
    void write_mol_gjf(const std::string &ofilename, int imol=0, bool write_connectivity=false) const;
    void get_backbone();
    void get_backbone_without_hydrogen();
    void get_intermolecule_connectivity_of_selected_part();
    void get_a_nice_molecule();
    void connection_search(std::vector<unsigned char> &visited, int i);
    void generate_monomer(int imol);
    void generate_dimer(int imol, int jmol, char shift_a=0, char shift_b=0, char shift_c=0);

};

#endif // __MOLECULES_BOX_HPP__

