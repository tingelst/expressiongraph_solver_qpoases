cmake_minimum_required(VERSION 2.8.3)
project(expressiongraph_solver_qpoases)

find_package(catkin  REQUIRED)
find_package(cmake_modules REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(catkin REQUIRED COMPONENTS expressiongraph expressiongraph_context)
include_directories( include ${catkin_INCLUDE_DIRS})

# external libraries:
SET(qpOASES qpOASES-3.0beta)

add_definitions( -D__NO_COPYRIGHT__ )


INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/external/qpOASES/${qpOASES}/include ${catkin_INCLUDE_DIRS})


add_library(${PROJECT_NAME}  src/qpoases_messages.cpp      
                             src/qpoases_solver.cpp 
                             )

target_link_libraries(${PROJECT_NAME} expressiongraph qpOASES ${catkin_LIBRARIES})

catkin_package(
  INCLUDE_DIRS include ${PROJECT_SOURCE_DIR}/external/qpOASES/${qpOASES}/include
  LIBRARIES ${PROJECT_NAME} expressiongraph qpOASES
  CATKIN_DEPENDS expressiongraph
  DEPENDS expressiongraph)

ADD_SUBDIRECTORY( external/qpOASES )
#message("${Red}${PROJECT_SOURCE_DIR}/external/qpOASES/${qpOASES}/include")



install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY external/qpOASES/${qpOASES}/include/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE)
 
install(DIRECTORY include/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE)
   
  
