cmake_minimum_required( VERSION 3.10 )
project(barcode_simulator)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# List of source files for the simulation executable
set( simulation_src
  src/utils.cpp
  src/cell.cpp
  src/simulationmain.cpp
  src/simulation.cpp
  src/migrationgraph.cpp
  src/clonetree.cpp
  src/basetree.cpp
)

include_directories( "${CMAKE_SOURCE_DIR}/lib/lemon/include" "src" )
link_directories( "${CMAKE_SOURCE_DIR}/lib/lemon/lib" )

set( CommonLibs emon )

add_executable( simulate ${simulation_src} )
target_link_libraries( simulate ${CommonLibs} )
