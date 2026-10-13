#ifndef OEVIBE_CORE_SCANGRID_H
#define OEVIBE_CORE_SCANGRID_H

#include <cstddef>
#include <vector>

#include "oevibe/Options.h"

namespace OEVIBE {
namespace detail {

/// Degrees. A step divides a range, and a range spans exactly 360 degrees, when they do within
/// this tolerance. It sits three orders below the 1e-3 degree constraint tolerance (spec 7.1), so
/// an accepted grid differs from the requested one by less than the optimizer can resolve, while
/// decimal steps that binary floating point cannot represent exactly, such as 0.1, still divide.
constexpr double SCAN_ANGLE_TOLERANCE = 1e-6;

/// The most points a scan grid may hold. A step too fine for this is an OptionsError rather than
/// an allocation that cannot succeed; at 20 optimizations per point, a million points is already
/// far beyond an interactive scan.
constexpr std::size_t MAX_SCAN_GRID_POINTS = 1000000;

/// One scanned dihedral's grid (spec 7.4). ``start``, ``end``, and ``step`` are as requested;
/// ``values`` are normalized to (-180, 180] and follow the range from ``start`` to ``end``.
struct ScanAxis {
    double start = -180.0;
    double end = 180.0;
    double step = 15.0;
    bool periodic = false;  ///< True when the range spans exactly 360 degrees.
    std::vector<double> values;
};

/// The Cartesian product of one or more axes, indexed row-major with the first axis varying
/// slowest (spec 7.4).
class ScanGrid {
public:
    /// :raises std::invalid_argument: Without axes, for an axis without values, or when the
    ///     product of the axis sizes exceeds MAX_SCAN_GRID_POINTS.
    explicit ScanGrid(std::vector<ScanAxis> axes);

    std::size_t Size() const;
    const std::vector<ScanAxis>& GetAxes() const;
    /// :returns: Degrees, one per axis.
    /// :raises std::out_of_range: For an index past the end.
    std::vector<double> Angles(std::size_t index) const;
    /// :returns: The points one step away along each axis, wrapping on periodic axes, ascending
    ///     and without repeats or ``index`` itself.
    /// :raises std::out_of_range: For an index past the end.
    std::vector<std::size_t> Neighbors(std::size_t index) const;
    /// Distances are angular differences wrapped to [0, 180] on every axis, because a dihedral
    /// is periodic even where the grid's adjacency is not; ties go to the lower index.
    /// :returns: The point nearest ``angles`` (degrees, one per axis).
    /// :raises std::invalid_argument: When ``angles`` does not have one value per axis.
    std::size_t Nearest(const std::vector<double>& angles) const;

private:
    std::vector<std::size_t> Positions(std::size_t index) const;
    std::size_t IndexOf(const std::vector<std::size_t>& positions) const;

    std::vector<ScanAxis> axes_;
    std::size_t size_ = 1;
};

/// :returns: The number of steps from ``start`` to ``end``.
/// :raises OptionsError: Unless all three are finite, ``step`` > 0, ``end`` > ``start``, the
///     range is at most 360 degrees, and ``step`` divides it, each within SCAN_ANGLE_TOLERANCE,
///     and there are at most MAX_SCAN_GRID_POINTS steps.
std::size_t scan_axis_intervals(double start, double end, double step);

/// Spec 7.4: a range spanning exactly 360 degrees gives the half-open grid start + step, ...,
/// end with periodic adjacency; any other range is inclusive at both ends and not periodic.
/// :raises OptionsError: As for ``scan_axis_intervals``.
ScanAxis build_scan_axis(double start, double end, double step);

/// :returns: One axis per dihedral, all sharing the options' range and step.
/// :raises OptionsError: When the options fail ``Validate()``.
ScanGrid build_scan_grid(const ScanOptions& options);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_SCANGRID_H
