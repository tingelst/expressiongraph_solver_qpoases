#SET(qpOASES qpOASES-3.0beta)
#INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/external/qpOASES/${qpOASES}/include)

# compile qpOASES libraries
FILE(GLOB SRC ${qpOASES}/src/*.cpp )
FILE(GLOB EXTRAS_SRC ${qpOASES}/src/extras/*.cpp )

add_definitions( -D__NO_COPYRIGHT__ )
if(NOT USE_ROSBUILD)
  add_library(qpOASES ${SRC} ${EXTRAS_SRC})
elseif()
  rosbuild_add_library(qpOASES ${SRC} ${EXTRAS_SRC})
endif()


