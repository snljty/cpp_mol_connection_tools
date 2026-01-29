@echo off
g++ -o get_gro_connectivity get_gro_connectivity.cpp -I C:\eigen-3.4.1\include\eigen3 -I C:\fmt-12.1.0\include -L C:\fmt-12.1.0\lib -l fmt -std=c++17 -O2 -g -DNDEBUG
