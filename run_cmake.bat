@echo off

set CMAKE_PREFIX_PATH=C:/eigen-3.4.1;C:/fmt-12.1.0
cmake .. -G "MinGW Makefiles" -D CMAKE_INSTALL_PREFIX=C:/get_gro_connectivity -LH
