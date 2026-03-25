# Library for chroma decoders

# For M_PI constant
add_compile_definitions(_USE_MATH_DEFINES)

add_library(lddecode-chroma STATIC
	decoder.cpp
	decoderpool.cpp
    comb.cpp
    componentframe.cpp
    framecanvas.cpp
    outputwriter.cpp
    palcolour.cpp
    sourcefield.cpp
    transformpal.cpp
    transformpal2d.cpp
    transformpal3d.cpp
	monodecoder.cpp
)

set(ORT_ROOT "/home/ethan/git/vhs-decode/tools/onnxruntime-linux-x64-1.23.2") 

target_include_directories(lddecode-chroma PUBLIC .)

target_link_libraries(lddecode-chroma PRIVATE Qt::Core ${FFTW_LIBRARY} lddecode-library)

target_include_directories(lddecode-chroma PRIVATE ${FFTW_INCLUDE_DIR})

# ld-chroma-decoder
if(UNIX)
    # Confirm the version of this edition and the version below (1.23.2 or 1.16.3)
    # Please confirm that the version number here matches the one you actually downloaded (1.23.2 or 1.16.3)
    # (CN) 请确认这里的版本号与您实际下载的一致 (1.23.2 或 1.16.3)

    link_directories(${ORT_ROOT}/lib)

    include_directories(${ORT_ROOT}/include)
endif()
add_executable(ld-chroma-decoder
    decoder.cpp
    decoderpool.cpp
    main.cpp
    monodecoder.cpp
    ntscdecoder.cpp
    paldecoder.cpp
)

set_target_properties(lddecode-chroma PROPERTIES AUTOMOC ON)

target_include_directories(ld-chroma-decoder PRIVATE ${FFTW_INCLUDE_DIR})

target_link_libraries(ld-chroma-decoder PRIVATE Qt::Core lddecode-library lddecode-chroma)

install(TARGETS ld-chroma-decoder)

# --- Add ONNX Runtime support ---
# (CN) --- 添加 ONNX Runtime 支持 ---

include_directories(${ORT_ROOT}/include)

link_directories(${ORT_ROOT}/lib)

# (Optional) Ensure the runtime can find the .so file
# On Linux, you can configure LD_LIBRARY_PATH, or you can configure RPATH in CMake
# (CN) (可选) 确保运行时能找到 .so 文件
# (CN) 在 Linux 上，你可能需要设置 LD_LIBRARY_PATH，或者在 CMake 中设置 RPATH
set_target_properties(lddecode-chroma PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ORT_ROOT}/lib
    NO_DEFAULT_PATH
)

target_link_libraries(lddecode-chroma PUBLIC ${ONNXRUNTIME_LIB})

install(
    DIRECTORY ${ORT_ROOT}/lib/
    DESTINATION lib
    FILES_MATCHING PATTERN "libonnxruntime.so*"
)