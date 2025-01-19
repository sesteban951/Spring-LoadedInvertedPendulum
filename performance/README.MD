## Dependencies
### yaml-cpp
```bash
https://github.com/jbeder/yaml-cpp
```
### Eigen 3.4.0
```bash
https://eigen.tuxfamily.org/index.php?title=Main_Page
```
See the ```CMakeLists.txt``` and modify it if necessary. 

## Running
Build the executables:
```bash
cd <path_to_Spring-LoadedInvertedPendulum>/performance/
mkdir build
cmake ..
make
```

Create a data folder to store some simulation data:
```bash
cd ..
mkdir data
```

Change values in ```config/config.yaml``` to play with the controller.

Run the executable:
```bash
cd build
./a
```

Plot the result in MATLAB by running the ```plot_data.m``` MATLAB script.

