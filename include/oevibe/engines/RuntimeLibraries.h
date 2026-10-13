#ifndef OEVIBE_ENGINES_RUNTIMELIBRARIES_H
#define OEVIBE_ENGINES_RUNTIMELIBRARIES_H

#include <string>

namespace OEVIBE {

/// Sets the Python prefix library directory, which runtime-loaded engines search after
/// ``$CONDA_PREFIX/lib`` (spec 8.3). The Python package passes its prefix's ``lib`` directory;
/// an empty string removes the directory from the search. An engine reads the value when it
/// first loads its library, so later changes do not affect an engine that has already loaded.
void set_python_library_directory(const std::string& directory);

/// :returns: The directory last passed to ``set_python_library_directory``, or an empty string.
std::string get_python_library_directory();

}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_RUNTIMELIBRARIES_H
