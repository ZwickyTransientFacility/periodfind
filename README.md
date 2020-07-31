# PeriodFind

A collection of CUDA-accelerated periodicity detection algorithms, with both C++ and Python APIs.

## Installing

Before attempting to install, ensure that CUDA is installed, and `nvcc` is added to your `PATH` variable, or it may not be found by CMake or the Python setup file.

### Python API

Ensure that `Cython` and `numpy` are both installed. Then, simply run:

```bash
python setup.py install
```

And periodfind should be installed!

### C++ API

First, ensure that CMake is installed, and that it is at least version `3.8`. Next, create a build directory for CMake to use, and `cd` into it:

```bash
mkdir cmakebuild
cd cmakebuild
```

Now, run CMake, and build the library:

```bash
cmake ..
make
```

Finally, install the package by running `make install` (may require super-user priveleges), which will install the library in `/usr/local/lib/` and the headers in `/usr/local/include/periodfind/` by default (on Linux, location will be different on other operating systems).
