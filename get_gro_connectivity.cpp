// Eigen is column-major by default !!!
// coordinates are 3 * n.
// box is in column sequence storaged, each column corresponds to a lattice vector.
// matrix[i, j] for point i to j where i is row and j is column.
#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <regex>
#include <algorithm>
#include <tuple>
#include <array>
#include <stdexcept>
#include <iomanip>

// parallel?
// #pragma omp parallel for collapse(2)

const int ncoords = 3;

typedef enum {coord_x, coord_y, coord_z} coord_direction;

static const std::vector<std::string> elements_names = {
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

static const std::vector<float> elements_van_der_Waals_radius = {
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

static const std::vector<float> elements_covalence_radius = {
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

void getline_check(std::ifstream &ifile, std::string &line);

std::vector<int> indices_str_to_list_from_0(const std::string &indices);

void indices_str_to_list_from_0(const std::string &indices, std::vector<int> &result);

std::string list_to_indices_str_from_1(const std::vector<int> &nums);

void list_to_indices_str_from_1(const std::vector<int> &nums, std::string &result);

bool is_box_orthorhombic(const Eigen::Matrix3f &box);

Eigen::MatrixXf get_dist_matrix_no_box(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2);

std::pair<Eigen::MatrixXf, Eigen::Vector<char, ncoords> > 
    get_dist_matrix_orthorhombic(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2, const Eigen::Matrix3f box);
    // return dist_matrix and shift_min

std::pair<Eigen::MatrixXf, Eigen::Vector<char, ncoords> > 
    get_dist_squared_matrix_orthorhombic(const Eigen::MatrixXf &coord1, const Eigen::MatrixXf &coord2, const Eigen::Matrix3f box);
    // return dist_squared_matrix and shift_min

void print_box(const Eigen::Matrix3f &box);

class Element_table {
public:
    Element_table();
    std::unordered_map<std::string, int> table;
};

class Gro_file {
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

    Gro_file() = default;
    Gro_file(const std::string &ifilename, int nmols=0);
    ~Gro_file() = default;
    void read_gro(const std::string &ifilename, int nmols=0);
    void set_selection(const std::string &selection_str);
    void set_selection(const std::vector<int> &selection_list);
    void set_selection_internal();
    void get_intramolecule_connectivity(int imol=0, float scaler_tol=1.15f);
    void write_mol_gjf(const std::string &ofilename, int imol=0, bool write_connectivity=false) const;
    std::vector<int> get_backbone() const;
    std::vector<int> get_backbone_without_hydrogen() const;
    void get_intermolecule_connectivity_of_selected_part();

};

int main(int argc, const char *argv[]) {
    /*
    std::ifstream ifile("elements.txt");
    if (! ifile) {
        std::ofstream ofile("elements.txt");
        for (int i = 0; i < gro_file.natoms_per_mol; ++ i) {
            ofile << gro_file.elements[i] << std::endl;
        }
        ofile.close();
    } else {
        ifile.close();
    }
    */

    Gro_file gro_file("../frame0.gro");

    gro_file.get_intramolecule_connectivity();
    // gro_file.write_mol_gjf("test.gjf", 0, true);
    std::vector<int> backbone_noH = gro_file.get_backbone_without_hydrogen();
    // std::cout << list_to_indices_str_from_1(backbone_noH) << std::endl;
    gro_file.set_selection(backbone_noH);

    gro_file.get_intermolecule_connectivity_of_selected_part();

    std::ofstream ofile("short_contact.txt");
    ofile << "# counts from 1" << std::endl;
    ofile << "# natoms    npairs    imol    jmol    nx    ny    nz" << std::endl;
    for (int imol = 0; imol < gro_file.nmols; ++ imol) {
        for (int jmol = imol + 1; jmol < gro_file.nmols; ++ jmol) {
            if (gro_file.num_intermolecule_short_contact_pairs(imol, jmol)) {
                ofile << 
                std::setw(3) << "" << std::setw(3) << gro_file.num_intermolecule_short_contact_atoms(imol, jmol) << 
                std::setw(7) << "" << std::setw(3) << gro_file.num_intermolecule_short_contact_pairs(imol, jmol) << 
                std::setw(7) << "" << std::setw(3) << imol + 1 << 
                std::setw(5) << "" << std::setw(3) << jmol + 1 << 
                std::setw(4) << "" << std::setw(2) << static_cast<int>(gro_file.shifts_min[coord_x](imol, jmol)) << 
                std::setw(4) << "" << std::setw(2) << static_cast<int>(gro_file.shifts_min[coord_y](imol, jmol)) << 
                std::setw(4) << "" << std::setw(2) << static_cast<int>(gro_file.shifts_min[coord_z](imol, jmol)) << 
                std::endl;
            }
        }
    }
    ofile.close();

    /*
    std::ofstream ofile2("connectivity.txt");
    for (int i = 0; i < gro_file.natoms_per_mol; ++ i) {
        for (int j = 0; j < gro_file.natoms_per_mol; ++ j) {
            if (j) ofile2 << ' ';
            ofile2 << static_cast<int>(gro_file.intramolecule_connectivity(i, j));
        }
        ofile2 << std::endl;
    }
    ofile2.close();
    */

    // gro_file.write_mol_gjf("test.gjf", 0, true);

    /*
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < ncoords; ++ i) {
        for (int j = 0; j < ncoords; ++ j) {
            if (j) std::cout << std::setw(4) << "";
            std::cout << std::setw(8) << gro_file.box(i, j);
        }
        std::cout << std::endl;
    }
    std::cout << std::defaultfloat;
    */

    return 0;
}

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
    for (int i = 1; i < nums.size(); ++ i) {
        if (nums[i] == b + 1) {
            ++ b;
        } else {
            if (a == b) {
                ret_list.emplace_back(std::to_string(a + 1));
            } else {
                ret_list.emplace_back(std::to_string(a + 1) + "-" + std::to_string(b + 1));
            }
            a = nums[i];
            b = a;
        }
    }
    // handle last
    if (a == b) {
        ret_list.emplace_back(std::to_string(a + 1));
    } else {
        ret_list.emplace_back(std::to_string(a + 1) + "-" + std::to_string(b + 1));
    }
    // join ret_list to ret
    std::string ret;
    if (! ret_list.empty()) {
        ret = ret_list[0];
    }
    for (int i = 1; i < ret_list.size(); ++ i) {
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
    for (int i = 1; i < nums.size(); ++ i) {
        if (nums[i] == b + 1) {
            ++ b;
        } else {
            if (a == b) {
                ret_list.emplace_back(std::to_string(a + 1));
            } else {
                ret_list.emplace_back(std::to_string(a + 1) + "-" + std::to_string(b + 1));
            }
            a = nums[i];
            b = a;
        }
    }
    // handle last
    if (a == b) {
        ret_list.emplace_back(std::to_string(a + 1));
    } else {
        ret_list.emplace_back(std::to_string(a + 1) + "-" + std::to_string(b + 1));
    }
    // join ret_list to ret
    if (! ret_list.empty()) {
        result = ret_list[0];
    }
    for (int i = 1; i < ret_list.size(); ++ i) {
        result += "," + ret_list[i];
    }
}

bool is_box_orthorhombic(const Eigen::Matrix3f &box) {
    /*
    if (box.rows() != ncoords || box.cols() != ncoords) {
        throw std::invalid_argument("box must be " + std::to_string(ncoords) + " by " + std::to_string(ncoords) + ".");
    }
    */
    return box(1, 0) == 0.f && box(2, 0) == 0.f && box(2, 1) == 0.f && 
           box(0, 1) < 1.E-6f && box(0, 2) < 1.E-6f && box(1, 2) < 1.E-6f;
}

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
    float min_dist = dist_matrix.row(0).minCoeff(& min_dist_pos);
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
    float min_dist = squared_dist_matrix.row(0).minCoeff(& min_dist_pos);
    squared_dist_matrix.resize(coord1.cols(), coord2.cols());
    Eigen::Vector<char, ncoords> shift_min = shift_real.col(min_dist_pos).cast<char>();
    return std::make_pair(squared_dist_matrix, shift_min);
}

void print_box(const Eigen::Matrix3f &box) {
    /*
    if (box.rows() != ncoords || box.cols() != ncoords) {
        throw std::invalid_argument("box must be " + std::to_string(ncoords) + " by " + std::to_string(ncoords) + ".");
    }
    */
    std::cout << std::fixed << std::setprecision(8);
    for (int i = 0; i < ncoords; ++ i) {
        std::cout   << std::setw(11) << box(coord_x, i) << 
            "    "  << std::setw(11) << box(coord_y, i) << 
            "    "  << std::setw(11) << box(coord_z, i) << std::endl;
    }
}

Element_table::Element_table() {
    for (int i = 0; i < elements_names.size(); ++ i) {
        table.insert({elements_names[i], i});
    }
}

Gro_file::Gro_file(const std::string &ifilename, int nmols) : element_table() {
    read_gro(ifilename, nmols);
}

void Gro_file::read_gro(const std::string &ifilename, int nmols) {
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
    if (elements.size() != natoms_per_mol) {
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
    if (getline(ifile, line)) {
        throw std::ios_base::failure("Extra line found.");
    }
    box = box_d.cast<float>();
    ifile.close();
}

void Gro_file::set_selection(const std::string &selection_str) {
    indices_str_to_list_from_0(selection_str, selection);
    set_selection_internal();
}

void Gro_file::set_selection(const std::vector<int> &selection_list) {
    selection = selection_list;
    set_selection_internal();
}

void Gro_file::set_selection_internal() {
    natoms_selected = selection.size();

    if (atomic_indices_selected.size() != natoms_selected) {
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

void Gro_file::get_intramolecule_connectivity(int imol, float scaler_tol) {
    // get distance matrix, no PBC considered
    Eigen::MatrixXf dist_matrix = get_dist_matrix_no_box(coordinates[imol], coordinates[imol]);

    intramolecule_connectivity = (dist_matrix.array() <= (atomic_covalence_radius_sum * scaler_tol).array()).cast<unsigned char>();
    intramolecule_connectivity.diagonal().setZero();
}

void Gro_file::get_intermolecule_connectivity_of_selected_part() {
    static std::pair<Eigen::MatrixXf, Eigen::Vector<char, ncoords> > current;
    static Eigen::MatrixXi short_contact;

    for (int jmol = 0; jmol < nmols; ++ jmol) {
        for (int imol = jmol + 1; imol < nmols; ++ imol) {
            std::cout << "\rhandling imol = " << std::setw(3) << imol << ", jmol = " << std::setw(3) << jmol << std::flush;
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
    std::cout << "\r" << std::setw(100) << "" << "\r" << std::flush;

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

void Gro_file::write_mol_gjf(const std::string &ofilename, int imol, bool write_connectivity) const {
    size_t pos = ofilename.rfind('.');
    if (pos == std::string::npos || ofilename.substr(pos, ofilename.length() - pos) != ".gjf") {
        throw std::invalid_argument("Suffix of a gjf file must be \".gjf\".");
    }

    std::ofstream ofile(ofilename);
    ofile << "%%chk=" << ofilename.substr(0, pos) << ".chk" << std::endl;
    if (write_connectivity) {
        ofile << "#P B3LYP/6-31G** EmpiricalDispersion=GD3BJ Geom=Connectivity" << std::endl;
    } else {
        ofile << "#P B3LYP/6-31G** EmpiricalDispersion=GD3BJ" << std::endl;
    }
    ofile << std::endl << "molecule " << imol + 1 << std::endl << std::endl;
    const int charge = 0, multiplicity = 1;
    ofile << ' ' << charge << ' ' << multiplicity << std::endl;
    ofile << std::fixed << std::setprecision(5);
    for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
        ofile << ' ' << std::left << std::setw(2) << elements[iatom] << 
            "    " << std::setw(10) << coordinates[imol](coord_x, iatom) << 
            "    " << std::setw(10) << coordinates[imol](coord_y, iatom) << 
            "    " << std::setw(10) << coordinates[imol](coord_z, iatom) << std::endl;
    }
    ofile << std::endl;
    if (write_connectivity){
        for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
            ofile << iatom + 1;
            for (int jatom = iatom + 1; jatom < natoms_per_mol; ++ jatom) {
                if (intramolecule_connectivity(iatom, jatom)) {
                    ofile << ' ' << jatom + 1 << ' ' << "1.";
                }
            }
            ofile << std::endl;
        }
        ofile << std::endl;
    }
    ofile << std::defaultfloat;
    ofile.close();
}

std::vector<int> Gro_file::get_backbone() const {
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
    std::vector<int> backbone;
    for (int iatom = 0; iatom < natoms_per_mol; ++ iatom) {
        if (! in_alkyl[iatom]) backbone.push_back(iatom);
    }
    return backbone;
}

std::vector<int> Gro_file::get_backbone_without_hydrogen() const {
    std::vector<int> backbone = get_backbone();
    std::vector<int> backbone_noH;
    backbone_noH.reserve(backbone.size());
    for (int iatom : backbone) {
        if (elements[iatom] != "H") backbone_noH.push_back(iatom);
    }
    return backbone_noH;
}
