#include "molecules_box.hpp"

void MoleculesBox::generate_monomer(int imol) {
    Eigen::Vector3f bond_vec;
    const float C_H_bond_len = 1.09105f; // sp3 C-H length in ethane at B3LYP-D3(BJ)/def2-TZVPP

    monomer.coordinates = coordinates[imol];

    monomer_methyl.coordinates.block(0, 0, ncoords, backbone.size()) = coordinates[imol](Eigen::all, backbone);
    int iatom_in_methyl = 0;
    for (size_t imethyl = 0; imethyl < alkyl_connection_site.size(); ++ imethyl) {
        int iatom = alkyl_connection_site[imethyl];
        monomer_methyl.coordinates.col(backbone.size() + iatom_in_methyl) = coordinates[imol].col(iatom);
        ++ iatom_in_methyl;
        for (size_t i = 0; i < alkyl_connection_site_connected_alkyl[imethyl].size(); ++ i) {
            int jatom = alkyl_connection_site_connected_alkyl[imethyl][i];
            bond_vec = coordinates[imol].col(jatom) - coordinates[imol].col(iatom);
            bond_vec /= bond_vec.norm() / C_H_bond_len;
            monomer_methyl.coordinates.col(backbone.size() + iatom_in_methyl + i) = coordinates[imol].col(iatom) + bond_vec;
        }
        iatom_in_methyl += alkyl_connection_site_connected_alkyl[imethyl].size();
    }

    for (size_t i = 0; i < backbone_noH.size(); ++ i) {
        monomer_backbone_no_hydrogen.coordinates.col(i) = coordinates[imol].col(backbone_noH[i]);
    }
}

void MoleculesBox::generate_dimer(int imol, int jmol, char shift_a, char shift_b, char shift_c) {
    Eigen::Vector3f bond_vec;
    const float C_H_bond_len = 1.09105f; // sp3 C-H length in ethane at B3LYP-D3(BJ)/def2-TZVPP

    dimer.coordinates.block(0, 0, ncoords, natoms_per_mol) = coordinates[imol];
    dimer.coordinates.block(0, natoms_per_mol, ncoords, natoms_per_mol) = coordinates[jmol];
    if (shift_a) dimer.coordinates.block(0, natoms_per_mol, ncoords, natoms_per_mol).colwise() -= box.col(0) * shift_a;
    if (shift_b) dimer.coordinates.block(0, natoms_per_mol, ncoords, natoms_per_mol).colwise() -= box.col(1) * shift_b;
    if (shift_c) dimer.coordinates.block(0, natoms_per_mol, ncoords, natoms_per_mol).colwise() -= box.col(2) * shift_c;

    dimer_methyl.coordinates.block(0, 0, ncoords, backbone.size()) = coordinates[imol](Eigen::all, backbone);
    dimer_methyl.coordinates.block(0, natoms_per_trimmed_mol, ncoords, backbone.size()) = coordinates[jmol](Eigen::all, backbone);
    int iatom_in_methyl = 0;
    for (size_t imethyl = 0; imethyl < alkyl_connection_site.size(); ++ imethyl) {
        int iatom = alkyl_connection_site[imethyl];
        dimer_methyl.coordinates.col(backbone.size() + iatom_in_methyl) = coordinates[imol].col(iatom);
        dimer_methyl.coordinates.col(backbone.size() + iatom_in_methyl + natoms_per_trimmed_mol) = coordinates[jmol].col(iatom);
        ++ iatom_in_methyl;
        for (size_t i = 0; i < alkyl_connection_site_connected_alkyl[imethyl].size(); ++ i) {
            int jatom = alkyl_connection_site_connected_alkyl[imethyl][i];
            bond_vec = coordinates[imol].col(jatom) - coordinates[imol].col(iatom);
            bond_vec /= bond_vec.norm() / C_H_bond_len;
            dimer_methyl.coordinates.col(backbone.size() + iatom_in_methyl + i) = coordinates[imol].col(iatom) + bond_vec;
            bond_vec = coordinates[jmol].col(jatom) - coordinates[jmol].col(iatom);
            bond_vec /= bond_vec.norm() / C_H_bond_len;
            dimer_methyl.coordinates.col(backbone.size() + iatom_in_methyl + i + natoms_per_trimmed_mol) = coordinates[jmol].col(iatom) + bond_vec;
        }
        iatom_in_methyl += alkyl_connection_site_connected_alkyl[imethyl].size();
    }
    if (shift_a) dimer_methyl.coordinates.block(0, natoms_per_trimmed_mol, ncoords, natoms_per_trimmed_mol).colwise() -= box.col(0) * shift_a;
    if (shift_b) dimer_methyl.coordinates.block(0, natoms_per_trimmed_mol, ncoords, natoms_per_trimmed_mol).colwise() -= box.col(1) * shift_b;
    if (shift_c) dimer_methyl.coordinates.block(0, natoms_per_trimmed_mol, ncoords, natoms_per_trimmed_mol).colwise() -= box.col(2) * shift_c;

    for (size_t i = 0; i < backbone_noH.size(); ++ i) {
        dimer_backbone_no_hydrogen.coordinates.col(i) = coordinates[imol].col(backbone_noH[i]);
        dimer_backbone_no_hydrogen.coordinates.col(i + backbone_noH.size()) = coordinates[jmol].col(backbone_noH[i]);
    }
    if (shift_a) dimer_backbone_no_hydrogen.coordinates.block(0, backbone_noH.size(), ncoords, backbone_noH.size()).colwise() -= box.col(0) * shift_a;
    if (shift_b) dimer_backbone_no_hydrogen.coordinates.block(0, backbone_noH.size(), ncoords, backbone_noH.size()).colwise() -= box.col(1) * shift_b;
    if (shift_c) dimer_backbone_no_hydrogen.coordinates.block(0, backbone_noH.size(), ncoords, backbone_noH.size()).colwise() -= box.col(2) * shift_c;
}
