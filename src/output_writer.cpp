#include "utils.hpp"
#include "molecules_box.hpp"

#include <fmt/format.h>
#include <fmt/os.h>

void MoleculesBox::write_mol_gjf(const std::string &ofilename, int imol, bool write_connectivity) const {
    size_t pos = ofilename.rfind('.');
    if (pos == std::string::npos || ofilename.substr(pos, ofilename.length() - pos) != ".gjf") {
        throw std::invalid_argument("Suffix of a gjf file must be \".gjf\".");
    }

    fmt::ostream ofile(fmt::output_file(ofilename));

    ofile.print("%chk={:s}.chk\n", ofilename.substr(0, pos));
    if (write_connectivity) {
        ofile.print("#P B3LYP/6-31G** EmpiricalDispersion=GD3BJ Geom=Connectivity\n");
    } else {
        ofile.print("#P B3LYP/6-31G** EmpiricalDispersion=GD3BJ\n");
    }
    ofile.print("\nmolecule {:d}\n\n", imol + 1);
    const int charge = 0, multiplicity = 1;
    ofile.print(" {:d} {:d}\n", charge, multiplicity);
    for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
        // " {:<2s}    {:13.8f}    {:13.8f}    {:13.8f}\n"
        ofile.print(" {:<2s}    {:7.2f}    {:7.2f}    {:7.2f}\n", 
            elements[iatom], 
            coordinates[imol](coord_x, iatom), 
            coordinates[imol](coord_y, iatom), 
            coordinates[imol](coord_z, iatom)
        );
    }
    ofile.print("\n");

    if (write_connectivity){
        for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
            ofile.print("{:d}", iatom + 1);
            for (int jatom = iatom + 1; jatom < natoms_per_mol; ++ jatom) {
                if (intramolecule_connectivity(iatom, jatom)) {
                    ofile.print(" {:d} 1.", jatom + 1);
                }
            }
            ofile.print("\n");
        }
        ofile.print("\n");
    }

    ofile.close();
}

void Molecule::write_gjf(const std::string &ofilename) {
    size_t pos = ofilename.rfind('.');
    if (pos == std::string::npos || ofilename.substr(pos, ofilename.length() - pos) != ".gjf") {
        throw std::invalid_argument("Suffix of a gjf file must be \".gjf\".");
    }

    fmt::ostream ofile(fmt::output_file(ofilename));

    ofile.print("%chk={:s}.chk\n", ofilename.substr(0, pos));
    ofile.print("#P B3LYP/6-31G** EmpiricalDispersion=GD3BJ\n");
    ofile.print("\n{:s}\n\n", ofilename.substr(0, pos));
    const int charge = 0, multiplicity = 1;
    ofile.print(" {:d} {:d}\n", charge, multiplicity);
    for (int iatom = 0; iatom < natoms; ++ iatom) {
        // " {:<2s}    {:13.8f}    {:13.8f}    {:13.8f}\n"
        ofile.print(" {:<2s}    {:7.2f}    {:7.2f}    {:7.2f}\n", 
            elements[iatom], 
            coordinates(coord_x, iatom), 
            coordinates(coord_y, iatom), 
            coordinates(coord_z, iatom)
        );
    }
    ofile.print("\n");
    ofile.close();
}

void print_box(const Eigen::Matrix3f &box) {
    /*
    if (box.rows() != ncoords || box.cols() != ncoords) {
        throw std::invalid_argument("box must be " + std::to_string(ncoords) + " by " + std::to_string(ncoords) + ".");
    }
    */
    for (int i = 0; i < ncoords; ++ i) {
        // "{:12.8f}    {:12.8f}    {:12.8f}\n"
        fmt::print("{:8.4f}    {:8.4f}    {:8.4f}\n", 
            box(coord_x, i), 
            box(coord_y, i), 
            box(coord_z, i)
        );
    }
}

