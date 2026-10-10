include(OpenEyeSWIG)

openeye_add_swig_module(
    NAME oevibe
    SWIG_FILE oevibe.i
    LINK_LIBS oevibe
    PYTHON_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/python/oevibe
    STABLE_ABI ${OEVIBE_USE_STABLE_ABI}
    SWIG_FLAGS "-doxygen"
    EXPECTED_LIB_VARS
        OECHEM_LIBRARY OESYSTEM_LIBRARY OEPLATFORM_LIBRARY
        OEMATH_LIBRARY OEZSTD_LIBRARY
    INIT_PY ${CMAKE_SOURCE_DIR}/python/oevibe/__init__.py
)
