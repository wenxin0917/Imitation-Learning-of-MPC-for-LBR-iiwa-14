# Install script for directory: /home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/src/iiwa14

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install" TYPE PROGRAM FILES "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/_setup_util.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install" TYPE PROGRAM FILES "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/env.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/setup.bash;/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install" TYPE FILE FILES
    "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/setup.bash"
    "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/local_setup.bash"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/setup.sh;/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install" TYPE FILE FILES
    "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/setup.sh"
    "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/local_setup.sh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/setup.zsh;/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install" TYPE FILE FILES
    "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/setup.zsh"
    "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/install" TYPE FILE FILES "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/.rosinstall")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/iiwa14.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/iiwa14/cmake" TYPE FILE FILES
    "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/iiwa14Config.cmake"
    "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/iiwa14Config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/iiwa14" TYPE FILE FILES "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/src/iiwa14/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/iiwa14" TYPE PROGRAM FILES "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/offline.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/iiwa14" TYPE PROGRAM FILES "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/catkin_generated/installspace/publish_circle_trajectory.py")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/gtest/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/wenxin/Desktop/SAIL/Imitation-Learning-of-MPC-for-LBR-iiwa-14/ros_visualization/build/iiwa14/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
