// Eigen is column-major by default !!!
// coordinates are 3 * n.
// box is in column sequence storaged, each column corresponds to a lattice vector.
// matrix[i, j] for point i to j where i is row and j is column.
#include "utils.hpp"
#include "molecules_box.hpp"

#include <fmt/format.h>
#include <fmt/os.h>

int main(int argc, const char *argv[]) {
    if (argc - 1 < 2 || argc - 1 > 3) {
        fmt::print("Usage: {:s} prefix frame_index [npairs_output_tol]\n", argv[0]);
        fmt::print("Will handle $prefix$frame.gro\n");
        return 0;
    }

    int iframe = -1;
    iframe = std::atoi(argv[2]);
    if (iframe < 0) {
        throw std::invalid_argument("Cannot read frame_index.");
    }

    const int npairs_tol_def = 8;
    int npairs_tol = -1;
    if (argc - 1 == 3) {
        npairs_tol = std::atoi(argv[3]);
    }
    if (npairs_tol < 0) npairs_tol = npairs_tol_def;

    std::string ifilename = fmt::format("{:s}{:d}.gro", argv[1], iframe);

    MoleculesBox molecules_box(ifilename);

    molecules_box.get_a_nice_molecule();
    molecules_box.get_intramolecule_connectivity(molecules_box.index_of_a_nice_molecule);
    molecules_box.get_backbone_without_hydrogen();
    molecules_box.set_selection(molecules_box.backbone_noH);

    for (int imonomer = 0; imonomer < molecules_box.nmols; ++ imonomer) {
        molecules_box.generate_monomer(imonomer);
        molecules_box.monomer.write_gjf(fmt::format("frame_{:03d}_monomer_{:03d}.gjf", iframe, imonomer + 1));
        molecules_box.monomer_methyl.write_gjf(fmt::format("frame_{:03d}_monomer_trimmed_{:03d}.gjf", iframe, imonomer + 1));
        molecules_box.monomer_backbone_no_hydrogen.write_gjf(fmt::format("frame_{:03d}_monomer_backbone_only_{:03d}.gjf", iframe, imonomer + 1));
    }

    molecules_box.get_intermolecule_connectivity_of_selected_part();

    std::string ofilename = fmt::format("frame_{:03d}_short_contact.txt", iframe);
    int idimer = 0;
    fmt::ostream ofile(fmt::output_file(ofilename));
    ofile.print("# counts from 1\n");
    ofile.print("# natoms    npairs    imol    jmol    nx    ny    nz\n");
    for (int imol = 0; imol < molecules_box.nmols; ++ imol) {
        for (int jmol = imol + 1; jmol < molecules_box.nmols; ++ jmol) {
            if (molecules_box.num_intermolecule_short_contact_pairs(imol, jmol) >= npairs_tol) {
                molecules_box.generate_dimer(imol, jmol, 
                    molecules_box.shifts_min[coord_x](imol, jmol), 
                    molecules_box.shifts_min[coord_y](imol, jmol), 
                    molecules_box.shifts_min[coord_z](imol, jmol)
                );
                molecules_box.dimer.write_gjf(fmt::format("frame_{:03d}_dimer_{:04d}.gjf", iframe, idimer + 1));
                molecules_box.dimer_methyl.write_gjf(fmt::format("frame_{:03d}_dimer_trimmed_{:04d}.gjf", iframe, idimer + 1));
                molecules_box.dimer_backbone_no_hydrogen.write_gjf(fmt::format("frame_{:03d}_dimer_backbone_only_{:04d}.gjf", iframe, idimer + 1));
            }
            if (molecules_box.num_intermolecule_short_contact_pairs(imol, jmol)) {
                ofile.print("{0:3s}{1:3d}{0:7s}{2:3d}{0:7s}{3:3d}{0:5s}{4:3d}{0:4s}{5:2d}{0:4s}{6:2d}{0:4s}{7:2d}\n", 
                    "", 
                    molecules_box.num_intermolecule_short_contact_atoms(imol, jmol), 
                    molecules_box.num_intermolecule_short_contact_pairs(imol, jmol), 
                    imol + 1, 
                    jmol + 1, 
                    static_cast<int>(molecules_box.shifts_min[coord_x](imol, jmol)), 
                    static_cast<int>(molecules_box.shifts_min[coord_y](imol, jmol)), 
                    static_cast<int>(molecules_box.shifts_min[coord_z](imol, jmol))
                );
                ++ idimer;
            }
        }
    }
    ofile.close();

    return 0;
}
