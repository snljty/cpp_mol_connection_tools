#include "utils.hpp"
#include "molecules_box.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <regex>

void MoleculesBox::read_gro(const std::string &ifilename, int nmols) {
    this->nmols = nmols;
    // check ifilename
    size_t pos = ifilename.rfind('.');
    if (pos == std::string::npos || ifilename.substr(pos, ifilename.length() - pos) != ".gro") {
        throw std::invalid_argument("suffix of input file name must be \".gro\".");
    }
    std::ifstream ifile(ifilename);
    if (! ifile) {
        throw std::invalid_argument("Cannot open \"" + ifilename + " for reading.");
    }

    // get optional timestamp and natoms (total)
    std::string line;
    std::istringstream line_iss;
    getline_check(ifile, line); // timestamp
    pos = line.find("t=");
    if (pos) {
        line_iss.str(line.substr(pos, line.length() - pos));
        line_iss >> timestamp; // may fail;
        line_iss.clear();
    }
    getline_check(ifile, line); // natoms
    line_iss.str(line);
    if (! (line_iss >> natoms)) {
        throw std::ios_base::failure("Cannot read amount of atoms.");
    }

    // get nmols and natoms_per_mol
    int nchars_each_coord = 8;
    if (nmols == 0) {
        // get amount of molecules;
        getline_check(ifile, line);
        pos = line.find('.', 20);
        nchars_each_coord = line.find('.', pos + 1) - pos;
        std::string last_resid_str = line.substr(0, 5);
        natoms_per_mol = 1;
        for (int iatom = 1; iatom < natoms; ++ iatom) {
            getline_check(ifile, line);
            if (line.compare(0, 5, last_resid_str, 0, 5) != 0) break;
            ++ natoms_per_mol;
        }
        if (natoms % natoms_per_mol != 0) {
            throw std::runtime_error("amount of atoms per molecule cannot divide total amount of atoms.");
        }
        nmols = natoms / natoms_per_mol;

    } else {
        if (natoms % nmols != 0) {
            throw std::runtime_error("amount of molecules cannot divide total amount of atoms."); 
        }
        natoms_per_mol = natoms / nmols;
        getline_check(ifile, line);
        pos = line.find('.', 20);
        nchars_each_coord = line.find('.', pos + 1) - pos;
    }
    this->nmols = nmols;

    // get elements
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    getline_check(ifile, line); // title
    getline_check(ifile, line); // natoms
    if (elements.size() != static_cast<size_t>(natoms_per_mol)) {
        elements.resize(natoms_per_mol);
        atomic_indices.resize(natoms_per_mol);
        atomic_van_der_Waals_radius.resize(natoms_per_mol);
        atomic_covalence_radius.resize(natoms_per_mol);
        atomic_covalence_radius_sum.resize(natoms_per_mol, natoms_per_mol);
        coordinates.resize(nmols);
        coordinates_selected.resize(nmols);
        num_intermolecule_short_contact_atoms.resize(nmols, nmols);
        num_intermolecule_short_contact_pairs.resize(nmols, nmols);
        shifts_min[coord_x].resize(nmols, nmols);
        shifts_min[coord_y].resize(nmols, nmols);
        shifts_min[coord_z].resize(nmols, nmols);
    }
    std::string current_element;
    std::regex element_pattern("\\s*([A-Z][A-Za-z]?)[0-9]*\\s*");
    std::smatch matches;
    for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
        getline_check(ifile, line);
        current_element = line.substr(10, 5);
        if (! std::regex_match(current_element, matches, element_pattern)) {
            throw std::runtime_error("Cannot get element of atom " + std::to_string(iatom + 1) + ".");
        }
        current_element = matches[1];
        if (current_element.length() >= 2) current_element[1] = std::tolower(current_element[1]);
        elements[iatom] = current_element;
        std::unordered_map<std::string, int>::iterator it = element_table.table.find(current_element);
        if (it != element_table.table.end()) {
            atomic_indices[iatom] = it->second;
        } else {
            std::cerr << "Warning: cannot recognize element of atom " << iatom + 1 << '.' << std::endl;
            atomic_indices[iatom] = 0;
        }
        atomic_van_der_Waals_radius[iatom] = elements_van_der_Waals_radius[atomic_indices[iatom]];
        atomic_covalence_radius[iatom] = elements_covalence_radius[atomic_indices[iatom]];
    }
    atomic_covalence_radius_sum = atomic_covalence_radius.replicate(1, atomic_covalence_radius.size()) + 
                                  atomic_covalence_radius.transpose().replicate(atomic_covalence_radius.size(), 1);

    // get coordinates
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    getline_check(ifile, line); // title
    getline_check(ifile, line); // natoms
    for (int imol = 0; imol < nmols; ++ imol) {
        if (coordinates[imol].rows() != ncoords || coordinates[imol].cols() != natoms_per_mol) {
            coordinates[imol].resize(ncoords, natoms_per_mol);
        }
        for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
            getline_check(ifile, line);
            coordinates[imol](coord_x, iatom) = std::stof(line.substr(20, nchars_each_coord));
            coordinates[imol](coord_y, iatom) = std::stof(line.substr(20 + nchars_each_coord , nchars_each_coord));
            coordinates[imol](coord_z, iatom) = std::stof(line.substr(20 + nchars_each_coord * 2, nchars_each_coord));
        }
        coordinates[imol] *= 10.f;
    }

    // get box
    getline_check(ifile, line);
    /*
    if (box.rows() != ncoords || box.cols() != ncoords) {
        box_d.resize(ncoords, ncoords);
        box.resize(ncoords, ncoords);
    }
    */
    box_d.setZero();
    line_iss.clear();
    line_iss.str(line);
    if (! (line_iss >> box_d(coord_x, coord_x) >> box_d(coord_y, coord_y) >> box_d(coord_z, coord_z))) {
        throw std::ios_base::failure("Cannot read diagonal of box_d.");
    }
    double tmp_box_component[6];
    if (line_iss >> tmp_box_component[0] >> tmp_box_component[1] >> tmp_box_component[2] >> 
                    tmp_box_component[3] >> tmp_box_component[4] >> tmp_box_component[5]) {
        box_d(1, 0) = tmp_box_component[0];
        box_d(2, 0) = tmp_box_component[1];
        box_d(0, 1) = tmp_box_component[2];
        box_d(2, 1) = tmp_box_component[3];
        box_d(0, 2) = tmp_box_component[4];
        box_d(1, 2) = tmp_box_component[5];
        if (box_d(1, 0) != 0. || box_d(2, 0) != 0. || box_d(2, 1) != 0.) {
            throw std::invalid_argument("Unsupported box.");
        }
    }
    box_d *= 10.;
    if (std::getline(ifile, line)) {
        throw std::ios_base::failure("Extra line found.");
    }
    box = box_d.cast<float>();
    ifile.close();

    monomer.resize(natoms_per_mol);
    dimer.resize(2 * natoms_per_mol);
    monomer.elements = elements;
    std::copy(elements.begin(), elements.end(), dimer.elements.begin());
    std::copy(elements.begin(), elements.end(), std::next(dimer.elements.begin(), elements.size()));
}
