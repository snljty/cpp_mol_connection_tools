#include "molecules_box.hpp"
#include "molecule.hpp"

#include <Eigen/Dense>

void MoleculesBox::get_backbone(){
    Eigen::Vector<unsigned char, Eigen::Dynamic> num_intramolecule_connections = intramolecule_connectivity.rowwise().sum();
    // get alkyl
    std::vector<int> alkyl;
    alkyl.reserve(natoms_per_mol);
    std::vector<bool> in_alkyl(natoms_per_mol, false);
    for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
        if (elements[iatom] == "C" && num_intramolecule_connections(iatom) == 4) {
            alkyl.push_back(iatom);
            in_alkyl[iatom] = true;
            for (int jatom = 0; jatom < natoms_per_mol; ++ jatom) {
                if ((elements[jatom] == "H" || elements[jatom] == "F") && intramolecule_connectivity(iatom, jatom)) {
                    alkyl.push_back(jatom);
                    in_alkyl[jatom] = true;
                }
            }
        }
    }
    std::sort(alkyl.begin(), alkyl.end());
    // get backbone
    backbone.clear();
    for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
        if (! in_alkyl[iatom]) backbone.push_back(iatom);
    }

    // get connection site of alkyl, and atoms in alkyl that connects to them.
    alkyl_connection_site.clear();
    alkyl_connection_site_connected_alkyl.clear();
    natoms_truncated_methyl_total = 0;
    for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
        if (in_alkyl[iatom]) {
            bool being_alkyl_connection_site = false;
            for (int jatom = 0; jatom < natoms_per_mol; ++ jatom) {
                if (! in_alkyl[jatom] && intramolecule_connectivity(iatom, jatom)) {
                    being_alkyl_connection_site = true;
                }
            }
            if (being_alkyl_connection_site) {
                alkyl_connection_site.push_back(iatom);
                alkyl_connection_site_connected_alkyl.emplace_back();
                for (int jatom = 0; jatom < natoms_per_mol; ++ jatom) {
                    if (in_alkyl[jatom] && intramolecule_connectivity(iatom, jatom)) {
                        alkyl_connection_site_connected_alkyl.back().push_back(jatom);
                    }
                }
                natoms_truncated_methyl_total += 1 + alkyl_connection_site_connected_alkyl.back().size();
            }
        }
    }
    natoms_per_trimmed_mol = backbone.size() + natoms_truncated_methyl_total;
    monomer_methyl.resize(natoms_per_trimmed_mol);
    dimer_methyl.resize(2 * natoms_per_trimmed_mol);
    for (size_t i = 0; i < backbone.size(); ++ i) {
        dimer_methyl.elements[i + natoms_per_trimmed_mol] = 
        dimer_methyl.elements[i] = 
        monomer_methyl.elements[i] = elements[backbone[i]];
    }
    int iatom_in_methyl = 0;
    for (size_t imethyl = 0; imethyl < alkyl_connection_site.size(); ++ imethyl) {
        int iatom = alkyl_connection_site[imethyl];
        dimer_methyl.elements[backbone.size() + iatom_in_methyl + natoms_per_trimmed_mol] = 
        dimer_methyl.elements[backbone.size() + iatom_in_methyl] = 
        monomer_methyl.elements[backbone.size() + iatom_in_methyl] = elements[iatom];
        ++ iatom_in_methyl;
        for (size_t i = 0; i < alkyl_connection_site_connected_alkyl[imethyl].size(); ++ i) {
            dimer_methyl.elements[backbone.size() + iatom_in_methyl + i + natoms_per_trimmed_mol] = 
            dimer_methyl.elements[backbone.size() + iatom_in_methyl + i] = 
            monomer_methyl.elements[backbone.size() + iatom_in_methyl + i] = "H";
        }
        iatom_in_methyl += alkyl_connection_site_connected_alkyl[imethyl].size();
    }
}

void MoleculesBox::get_backbone_without_hydrogen() {
    if (backbone.size() == 0) get_backbone();
    backbone_noH.clear();
    backbone_noH.reserve(backbone.size());
    for (int iatom : backbone) {
        if (elements[iatom] != "H") backbone_noH.push_back(iatom);
    }

    monomer_backbone_no_hydrogen.resize(backbone_noH.size());
    dimer_backbone_no_hydrogen.resize(2 * backbone_noH.size());
    for (size_t i = 0; i < backbone_noH.size(); ++ i) {
        dimer_backbone_no_hydrogen.elements[backbone_noH.size() + i] = 
        dimer_backbone_no_hydrogen.elements[i] = 
        monomer_backbone_no_hydrogen.elements[i] = elements[backbone_noH[i]];
    }
}

MoleculesBox::MoleculesBox(const std::string &ifilename, int nmols) : element_table() {
    read_gro(ifilename, nmols);
}


void Molecule::resize(int natoms) {
    this->natoms = natoms;
    elements.resize(natoms);
    coordinates.resize(ncoords, natoms);
}

