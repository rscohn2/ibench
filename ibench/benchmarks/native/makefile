ICC=/localdisk/psxe_16/compilers_and_libraries_2016.2.181/linux/bin/intel64/icc

native: native.cpp
	$(ICC) -O2 -std=c++11 -mkl native.cpp -o native

native_deb: native.cpp
	$(ICC) -O0 -g -std=c++11 -mkl native.cpp -o native_deb
