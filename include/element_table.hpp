#pragma once
#ifndef __ELEMENT_TABLE_HPP__
#define __ELEMENT_TABLE_HPP__

#include <string>
#include <unordered_map>

class Element_table {
public:
    Element_table();
    std::unordered_map<std::string, int> table;
};

#endif // __ELEMENT_TABLE_HPP__

