cmake_minimum_required(VERSION 3.5)
project(expressiongraph_solver_qpoases)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(orocos_kdl REQUIRED)
find_package(expressiongraph REQUIRED)
find_package(expressiongraph_context REQUIRED)

set(qpOASES qpOASES-3.0beta)
add_subdirectory(external/qpOASES)

add_library(${PROJECT_NAME} SHARED src/qpoases_messages.cpp
                                   src/qpoases_solver.cpp)

target_include_directories(${PROJECT_NAME}
                           PRIVATE include external/qpOASES/${qpOASES}/include)

target_link_libraries(${PROJECT_NAME} qpOASES)

ament_target_dependencies(${PROJECT_NAME}
                          orocos_kdl
                          expressiongraph
                          expressiongraph_context
                          )


install(DIRECTORY include/ DESTINATION include)

install(DIRECTORY external/qpOASES/${qpOASES}/include/
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

ament_export_include_directories(include)

ament_export_libraries(${PROJECT_NAME})

ament_package()
