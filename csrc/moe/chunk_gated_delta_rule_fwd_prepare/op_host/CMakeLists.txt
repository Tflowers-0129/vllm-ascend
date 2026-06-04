set(CURRENT_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel")
get_filename_component(KERNEL_DIR_ABS ${KERNEL_DIR} ABSOLUTE)

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_fwd_prepare_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwdPrepare
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -I${CURRENT_CMAKE_DIR}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE chunk_gated_delta_rule_fwd_prepare ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
