add_definitions(-D__NO_COPYRIGHT__)

file(GLOB SRC ${qpOASES}/src/*.cpp)
file(GLOB EXTRAS_SRC ${qpOASES}/src/extras/*.cpp)

add_library(qpOASES SHARED ${SRC} ${EXTRAS_SRC})
target_include_directories(qpOASES PRIVATE ${qpOASES}/include)

install(DIRECTORY ${qpOASES}/include/ DESTINATION include)

install(TARGETS qpOASES
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
