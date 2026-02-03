#include "utils.hpp"

#include "element_table.hpp"

#include <cmath>

#include <fmt/format.h>

const std::vector<std::string> elements_names = {
    "",
    "H" , "He", 
    "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne",
    "Na", "Mg", "Al", "Si", "P" , "S" , "Cl", "Ar",
    "K" , "Ca", "Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I" , "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", 
    "Hf", "Ta", "W" , "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U" , "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", 
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nb", "Fl", "Mc", "Lv", "Ts", "Og"
};

constexpr std::array<float, max_element_num> elements_van_der_Waals_radius = {
    0.3f, 
    1.20f, 1.43f, 2.12f, 1.98f, 1.91f, 1.77f,
    1.66f, 1.50f, 1.46f, 1.58f, 2.50f, 2.51f,
    2.25f, 2.19f, 1.90f, 1.89f, 1.82f, 1.83f,
    2.73f, 2.62f, 2.58f, 2.46f, 2.42f, 2.45f,
    2.45f, 2.44f, 2.40f, 2.40f, 2.38f, 2.39f,
    2.32f, 2.29f, 1.88f, 1.82f, 1.86f, 2.25f,
    3.21f, 2.84f, 2.75f, 2.52f, 2.56f, 2.45f,
    2.44f, 2.46f, 2.44f, 2.15f, 2.53f, 2.49f,
    2.43f, 2.42f, 2.47f, 1.99f, 2.04f, 2.06f,
    3.48f, 3.03f, 2.98f, 2.88f, 2.92f, 2.95f,
    2.00f, 2.90f, 2.87f, 2.83f, 2.79f, 2.87f,
    2.81f, 2.83f, 2.79f, 2.80f, 2.74f, 2.63f,
    2.53f, 2.57f, 2.49f, 2.48f, 2.41f, 2.29f,
    2.32f, 2.45f, 2.47f, 2.60f, 2.54f, 2.00f,
    2.00f, 2.00f, 2.00f, 2.00f, 2.80f, 2.93f,
    2.88f, 2.71f, 2.82f, 2.81f, 2.83f, 3.05f,
    3.40f, 3.05f, 2.70f, 2.00f, 2.00f, 2.00f,
    2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f,
    2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 
    2.00f, 2.00f, 2.00f, 2.00f
};

constexpr std::array<float, max_element_num> elements_covalence_radius = {
    0.1f, 
    0.31f, 0.28f, 1.28f, 0.96f, 0.84f, 0.76f,
    0.71f, 0.66f, 0.57f, 0.58f, 1.66f, 1.41f,
    1.21f, 1.11f, 1.07f, 1.05f, 1.02f, 1.06f,
    2.03f, 1.76f, 1.70f, 1.60f, 1.53f, 1.39f,
    1.39f, 1.32f, 1.26f, 1.24f, 1.32f, 1.22f,
    1.22f, 1.20f, 1.19f, 1.20f, 1.20f, 1.16f,
    2.20f, 1.95f, 1.90f, 1.75f, 1.64f, 1.54f,
    1.47f, 1.46f, 1.42f, 1.39f, 1.45f, 1.44f,
    1.42f, 1.39f, 1.39f, 1.38f, 1.39f, 1.40f,
    2.44f, 2.15f, 2.07f, 2.04f, 2.03f, 2.01f,
    1.99f, 1.98f, 1.98f, 1.96f, 1.94f, 1.92f,
    1.92f, 1.89f, 1.90f, 1.87f, 1.87f, 1.75f,
    1.70f, 1.62f, 1.51f, 1.44f, 1.41f, 1.36f,
    1.36f, 1.32f, 1.45f, 1.46f, 1.48f, 1.40f,
    1.50f, 1.50f, 2.60f, 2.21f, 2.15f, 2.06f,
    2.00f, 1.96f, 1.90f, 1.87f, 1.80f, 1.69f,
    1.68f, 1.68f, 1.65f, 1.67f, 1.73f, 1.76f,
    1.61f, 1.57f, 1.49f, 1.43f, 1.41f, 1.34f,
    1.29f, 1.28f, 1.21f, 1.22f, 1.50f, 1.50f, 
    1.50f, 1.50f, 1.50f, 1.50f
};

void getline_check(std::ifstream &ifile, std::string &line) {
    if (! getline(ifile, line)) {
        throw std::ios_base::failure("Cannot read a line.");
    }
}

std::vector<int> indices_str_to_list_from_0(const std::string &indices) {
    std::vector<int> ret;
    std::istringstream indices_iss(indices), current_iss;
    std::string current_str;
    size_t pos;
    int a, b;
    while (getline(indices_iss, current_str, ',')) {
        current_iss.clear();
        pos = current_str.find('-');
        if (pos != std::string::npos) {
            current_str[pos] = ' ';
            current_iss.str(current_str);
            current_iss >> a >> b;
            for (int i = a; i <= b; ++ i) {
                ret.push_back(i - 1);
            }
        } else {
            current_iss.str(current_str);
            current_iss >> a;
            ret.push_back(a - 1);
        }
    }
    return ret;
}

void indices_str_to_list_from_0(const std::string &indices, std::vector<int> &result) {
    std::istringstream indices_iss(indices), current_iss;
    std::string current_str;
    size_t pos;
    int a, b;
    result.clear();
    while (getline(indices_iss, current_str, ',')) {
        current_iss.clear();
        pos = current_str.find('-');
        if (pos != std::string::npos) {
            current_str[pos] = ' ';
            current_iss.str(current_str);
            current_iss >> a >> b;
            for (int i = a; i <= b; ++ i) {
                result.push_back(i - 1);
            }
        } else {
            current_iss.str(current_str);
            current_iss >> a;
            result.push_back(a - 1);
        }
    }
}

std::string list_to_indices_str_from_1(const std::vector<int> &nums) {
    if (nums.empty()) return "";
    std::vector<std::string> ret_list;
    int a, b;
    // handle first
    a = nums[0];
    b = a;
    // handle second to the one before last
    for (size_t i = 1; i < nums.size(); ++ i) {
        if (nums[i] == b + 1) {
            ++ b;
        } else {
            if (a == b) {
                ret_list.emplace_back(fmt::format("{:d}", a + 1));
            } else {
                ret_list.emplace_back(fmt::format("{:d}-{:d}", a + 1, b + 1));
            }
            a = nums[i];
            b = a;
        }
    }
    // handle last
    if (a == b) {
        ret_list.emplace_back(fmt::format("{:d}", a + 1));
    } else {
        ret_list.emplace_back(fmt::format("{:d}-{:d}", a + 1, b + 1));
    }
    // join ret_list to ret
    std::string ret;
    if (! ret_list.empty()) {
        ret = ret_list[0];
    }
    for (size_t i = 1; i < ret_list.size(); ++ i) {
        ret += "," + ret_list[i];
    }
    return ret;
}

void list_to_indices_str_from_1(const std::vector<int> &nums, std::string &result) {
    result = "";
    if (nums.empty()) return;
    std::vector<std::string> ret_list;
    int a, b;
    // handle first
    a = nums[0];
    b = a;
    // handle second to the one before last
    for (size_t i = 1; i < nums.size(); ++ i) {
        if (nums[i] == b + 1) {
            ++ b;
        } else {
            if (a == b) {
                ret_list.emplace_back(fmt::format("{:d}", a + 1));
            } else {
                ret_list.emplace_back(fmt::format("{:d}-{:d}", a + 1, b + 1));
            }
            a = nums[i];
            b = a;
        }
    }
    // handle last
    if (a == b) {
        ret_list.emplace_back(fmt::format("{:d}", a + 1));
    } else {
        ret_list.emplace_back(fmt::format("{:d}-{:d}", a + 1, b + 1));
    }
    // join ret_list to ret
    if (! ret_list.empty()) {
        result = ret_list[0];
    }
    for (size_t i = 1; i < ret_list.size(); ++ i) {
        result += "," + ret_list[i];
    }
}

bool is_box_orthorhombic(const Eigen::Matrix3f &box) {
    if (box.rows() != ncoords || box.cols() != ncoords) {
        throw std::invalid_argument(fmt::format("box must be {:d} by {:d}.", ncoords, ncoords));
    }
    return box(1, 0) == 0.f && box(2, 0) == 0.f && box(2, 1) == 0.f && 
           box(0, 1) < __FLT_EPSILON__ && box(0, 2) < __FLT_EPSILON__ && box(1, 2) < __FLT_EPSILON__;
}

Element_table::Element_table() {
    for (size_t i = 0; i < elements_names.size(); ++ i) {
        table.insert({elements_names[i], i});
    }
}

