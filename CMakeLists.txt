cmake_minimum_required(VERSION 2.8)

project(mvm21)

option(USE_MPI "Compile with MPI support" ON)
option(USE_OMP "Compile with OpenMP support" ON)

unset(USE_OMP CACHE)

add_executable(main main.cpp)

if(USE_MPI)
    message("Dealing with MPI")
    find_package(MPI REQUIRED)
    if(NOT MPI_FOUND)
        message("MPI not found!")
    endif()

    message((${MPI_INCLUDE_PATH}))
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(main ${MPI_CXX_LIBRARIES})

    if(MPI_LINK_FLAGS)
        set_target_properties(main PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()

    add_definitions(-DUSE_MPI)
endif()

if(USE_OMP)
    message("Dealing with OpenMP")
    add_definitions(-DUSE_OMP)
    target_link_libraries(main -fopenmp)
endif()
