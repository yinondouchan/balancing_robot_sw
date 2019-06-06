cmake_minimum_required(VERSION 2.8)
project( follow_me )
find_package( OpenCV REQUIRED )

include_directories(./include)

add_executable( follow_me arduino_serial_comm.cpp follow_me.cpp detect_markers.cpp)
target_link_libraries( follow_me ${OpenCV_LIBS} )
