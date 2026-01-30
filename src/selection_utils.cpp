#include "molecules_box.hpp"

void MoleculesBox::set_selection(const std::string &selection_str) {
    indices_str_to_list_from_0(selection_str, selection);
    set_selection_internal();
}

void MoleculesBox::set_selection(const std::vector<int> &selection_list) {
    selection = selection_list;
    set_selection_internal();
}

void MoleculesBox::set_selection_internal() {
    natoms_selected = selection.size();

    if (atomic_indices_selected.size() != static_cast<size_t>(natoms_selected)) {
        atomic_indices_selected.resize(natoms_selected);
        atomic_van_der_Waals_radius_selected.resize(natoms_selected);
        atomic_van_der_Waals_radius_selected_sum.resize(natoms_selected, natoms_selected);
        atomic_van_der_Waals_radius_selected_sum_squared.resize(natoms_selected, natoms_selected);
        for (int imol = 0; imol < nmols; ++ imol) {
            coordinates_selected[imol].resize(ncoords, natoms_selected);
        }
    }

    for (int iselect = 0; iselect < natoms_selected; ++ iselect) {
        int iatom = selection[iselect];
        atomic_indices_selected[iselect] = atomic_indices[iatom];
        atomic_van_der_Waals_radius_selected[iselect] = atomic_van_der_Waals_radius[iatom];
        for (int imol = 0; imol < nmols; ++ imol) {
            coordinates_selected[imol].col(iselect) = coordinates[imol].col(iatom);
        }
    }
    atomic_van_der_Waals_radius_selected_sum = 
        atomic_van_der_Waals_radius_selected.replicate(1, natoms_selected) + 
        atomic_van_der_Waals_radius_selected.transpose().replicate(natoms_selected, 1);
    atomic_van_der_Waals_radius_selected_sum_squared = atomic_van_der_Waals_radius_selected_sum.array().square();
}
