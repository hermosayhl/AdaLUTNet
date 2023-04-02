#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "libcpufeatures" for configuration "RELEASE"
set_property(TARGET libcpufeatures APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libcpufeatures PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/libcpufeatures.a"
  )

list(APPEND _cmake_import_check_targets libcpufeatures )
list(APPEND _cmake_import_check_files_for_libcpufeatures "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/libcpufeatures.a" )

# Import target "libjpeg-turbo" for configuration "RELEASE"
set_property(TARGET libjpeg-turbo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libjpeg-turbo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/liblibjpeg-turbo.a"
  )

list(APPEND _cmake_import_check_targets libjpeg-turbo )
list(APPEND _cmake_import_check_files_for_libjpeg-turbo "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/liblibjpeg-turbo.a" )

# Import target "libtiff" for configuration "RELEASE"
set_property(TARGET libtiff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libtiff PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/liblibtiff.a"
  )

list(APPEND _cmake_import_check_targets libtiff )
list(APPEND _cmake_import_check_files_for_libtiff "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/liblibtiff.a" )

# Import target "libopenjp2" for configuration "RELEASE"
set_property(TARGET libopenjp2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libopenjp2 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/liblibopenjp2.a"
  )

list(APPEND _cmake_import_check_targets libopenjp2 )
list(APPEND _cmake_import_check_files_for_libopenjp2 "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/liblibopenjp2.a" )

# Import target "libpng" for configuration "RELEASE"
set_property(TARGET libpng APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libpng PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/liblibpng.a"
  )

list(APPEND _cmake_import_check_targets libpng )
list(APPEND _cmake_import_check_files_for_libpng "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/liblibpng.a" )

# Import target "tegra_hal" for configuration "RELEASE"
set_property(TARGET tegra_hal APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tegra_hal PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/libtegra_hal.a"
  )

list(APPEND _cmake_import_check_targets tegra_hal )
list(APPEND _cmake_import_check_files_for_tegra_hal "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/libtegra_hal.a" )

# Import target "ade" for configuration "RELEASE"
set_property(TARGET ade APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ade PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/libade.a"
  )

list(APPEND _cmake_import_check_targets ade )
list(APPEND _cmake_import_check_files_for_ade "${_IMPORT_PREFIX}/sdk/native/3rdparty/libs/armeabi-v7a/libade.a" )

# Import target "opencv_core" for configuration "RELEASE"
set_property(TARGET opencv_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/staticlibs/armeabi-v7a/libopencv_core.a"
  )

list(APPEND _cmake_import_check_targets opencv_core )
list(APPEND _cmake_import_check_files_for_opencv_core "${_IMPORT_PREFIX}/sdk/native/staticlibs/armeabi-v7a/libopencv_core.a" )

# Import target "opencv_imgproc" for configuration "RELEASE"
set_property(TARGET opencv_imgproc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_imgproc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/staticlibs/armeabi-v7a/libopencv_imgproc.a"
  )

list(APPEND _cmake_import_check_targets opencv_imgproc )
list(APPEND _cmake_import_check_files_for_opencv_imgproc "${_IMPORT_PREFIX}/sdk/native/staticlibs/armeabi-v7a/libopencv_imgproc.a" )

# Import target "opencv_highgui" for configuration "RELEASE"
set_property(TARGET opencv_highgui APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_highgui PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/sdk/native/staticlibs/armeabi-v7a/libopencv_highgui.a"
  )

list(APPEND _cmake_import_check_targets opencv_highgui )
list(APPEND _cmake_import_check_files_for_opencv_highgui "${_IMPORT_PREFIX}/sdk/native/staticlibs/armeabi-v7a/libopencv_highgui.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
