include(OpenEyeSWIG)

# oevibe.i %includes these headers, so an edit to any of them must re-run SWIG. Later tasks
# append each header they wrap.
set(_OEVIBE_SWIG_DEPENDS
    ${CMAKE_SOURCE_DIR}/include/oevibe/oevibe.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Units.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Method.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Preset.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Solvent.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Property.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Constraints.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Options.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Result.h
    ${CMAKE_SOURCE_DIR}/include/oevibe/Calculation.h
)
set_source_files_properties(oevibe.i PROPERTIES DEPENDS "${_OEVIBE_SWIG_DEPENDS}")

openeye_add_swig_module(
    NAME oevibe
    SWIG_FILE oevibe.i
    LINK_LIBS oevibe
    PYTHON_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/python/oevibe
    STABLE_ABI ${OEVIBE_USE_STABLE_ABI}
    SWIG_FLAGS "-doxygen" "-I${CMAKE_SOURCE_DIR}/include"
    EXPECTED_LIB_VARS
        OECHEM_LIBRARY OESYSTEM_LIBRARY OEPLATFORM_LIBRARY
        OEMATH_LIBRARY OEZSTD_LIBRARY
        OEBIO_LIBRARY OEGRID_LIBRARY OESZYBKI_LIBRARY OEQUACPAC_LIBRARY OEFF_LIBRARY
        OEAMBER_LIBRARY OESHEFFIELD_LIBRARY OEAM1BCC_LIBRARY OEAM1_LIBRARY OESMIRNOFF_LIBRARY
        OEMMFF_LIBRARY OESPICOLI_LIBRARY OEZAP_LIBRARY OESHAPE_LIBRARY OEHERMITE_LIBRARY
        OEMOLPOTENTIAL_LIBRARY OEOPT_LIBRARY
    INIT_PY ${CMAKE_SOURCE_DIR}/python/oevibe/__init__.py
)
