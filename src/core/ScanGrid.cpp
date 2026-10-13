#include "core/ScanGrid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/WriteBack.h"
#include "oevibe/Constraints.h"
#include "oevibe/Error.h"

namespace OEVIBE {
namespace detail {

namespace {

// Adding 0.0 turns -0 into 0, so an angle never prints as -0.000000.
double normalized(double degrees) { return normalize_dihedral_degrees(degrees) + 0.0; }

}  // namespace

ScanGrid::ScanGrid(std::vector<ScanAxis> axes) : axes_(std::move(axes)) {
    if (axes_.empty()) {
        throw std::invalid_argument("a scan grid needs at least one axis");
    }
    for (const ScanAxis& axis : axes_) {
        if (axis.values.empty()) {
            throw std::invalid_argument("a scan grid axis needs at least one value");
        }
        // Dividing first keeps the product from overflowing before it is compared.
        if (axis.values.size() > MAX_SCAN_GRID_POINTS / size_) {
            throw std::invalid_argument("a scan grid holds at most " +
                                        std::to_string(MAX_SCAN_GRID_POINTS) + " points");
        }
        size_ *= axis.values.size();
    }
}

std::size_t ScanGrid::Size() const { return size_; }

const std::vector<ScanAxis>& ScanGrid::GetAxes() const { return axes_; }

std::vector<std::size_t> ScanGrid::Positions(std::size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("scan grid index " + std::to_string(index) + " is out of range");
    }
    std::vector<std::size_t> positions(axes_.size());
    for (std::size_t k = axes_.size(); k-- > 0;) {
        const std::size_t count = axes_[k].values.size();
        positions[k] = index % count;
        index /= count;
    }
    return positions;
}

std::size_t ScanGrid::IndexOf(const std::vector<std::size_t>& positions) const {
    std::size_t index = 0;
    for (std::size_t k = 0; k < axes_.size(); ++k) {
        index = index * axes_[k].values.size() + positions[k];
    }
    return index;
}

std::vector<double> ScanGrid::Angles(std::size_t index) const {
    const std::vector<std::size_t> positions = Positions(index);
    std::vector<double> angles(axes_.size());
    for (std::size_t k = 0; k < axes_.size(); ++k) {
        angles[k] = axes_[k].values[positions[k]];
    }
    return angles;
}

std::vector<std::size_t> ScanGrid::Neighbors(std::size_t index) const {
    const std::vector<std::size_t> positions = Positions(index);
    std::vector<std::size_t> neighbors;
    for (std::size_t k = 0; k < axes_.size(); ++k) {
        const std::size_t count = axes_[k].values.size();
        std::vector<std::size_t> shifted = positions;
        if (positions[k] > 0) {
            shifted[k] = positions[k] - 1;
            neighbors.push_back(IndexOf(shifted));
        } else if (axes_[k].periodic) {
            shifted[k] = count - 1;
            neighbors.push_back(IndexOf(shifted));
        }
        if (positions[k] + 1 < count) {
            shifted[k] = positions[k] + 1;
            neighbors.push_back(IndexOf(shifted));
        } else if (axes_[k].periodic) {
            shifted[k] = 0;
            neighbors.push_back(IndexOf(shifted));
        }
    }
    // A periodic axis with one or two values wraps onto the point itself or onto one neighbor
    // twice.
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), index), neighbors.end());
    return neighbors;
}

std::size_t ScanGrid::Nearest(const std::vector<double>& angles) const {
    if (angles.size() != axes_.size()) {
        throw std::invalid_argument("expected " + std::to_string(axes_.size()) + " angles, got " +
                                    std::to_string(angles.size()));
    }
    // The grid is a Cartesian product, so the nearest point is the nearest value on each axis.
    std::vector<std::size_t> positions(axes_.size(), 0);
    for (std::size_t k = 0; k < axes_.size(); ++k) {
        const std::vector<double>& values = axes_[k].values;
        const auto distance = [&](std::size_t p) {
            return std::abs(normalize_dihedral_degrees(angles[k] - values[p]));
        };
        double best = distance(0);
        for (std::size_t p = 1; p < values.size(); ++p) {
            best = std::min(best, distance(p));
        }
        // A midpoint between two values is equidistant only to within rounding: for a 0.1 step
        // the computed distances differ in the last bit, and a strict comparison would pick
        // whichever side rounds lower. The grid already treats SCAN_ANGLE_TOLERANCE as exact
        // when accepting a step, so distances that close are ties and the lower index wins.
        for (std::size_t p = 0; p < values.size(); ++p) {
            if (distance(p) <= best + SCAN_ANGLE_TOLERANCE) {
                positions[k] = p;
                break;
            }
        }
    }
    return IndexOf(positions);
}

std::size_t scan_axis_intervals(double start, double end, double step) {
    if (!std::isfinite(start) || !std::isfinite(end)) {
        throw OptionsError("scan range ends must be finite");
    }
    if (!std::isfinite(step) || step <= 0.0) {
        throw OptionsError("scan step must be finite and greater than 0, got " +
                           format_value(step));
    }
    const double span = end - start;
    if (span <= 0.0) {
        throw OptionsError("scan range end " + format_value(end) + " must exceed its start " +
                           format_value(start));
    }
    if (span > 360.0 + SCAN_ANGLE_TOLERANCE) {
        throw OptionsError("scan range must not exceed 360 degrees, got " + format_value(span));
    }
    const double intervals = std::round(span / step);
    if (intervals < 1.0 || std::abs(intervals * step - span) > SCAN_ANGLE_TOLERANCE) {
        throw OptionsError("scan step " + format_value(step) + " does not divide the range of " +
                           format_value(span) + " degrees");
    }
    // Converting a count beyond std::size_t is undefined, so the bound is checked on the double.
    if (intervals > static_cast<double>(MAX_SCAN_GRID_POINTS)) {
        throw OptionsError("scan step " + format_value(step) + " is too fine for the range of " +
                           format_value(span) + " degrees: a scan grid holds at most " +
                           std::to_string(MAX_SCAN_GRID_POINTS) + " points");
    }
    return static_cast<std::size_t>(intervals);
}

ScanAxis build_scan_axis(double start, double end, double step) {
    const std::size_t intervals = scan_axis_intervals(start, end, step);
    ScanAxis axis;
    axis.start = start;
    axis.end = end;
    axis.step = step;
    axis.periodic = std::abs(end - start - 360.0) <= SCAN_ANGLE_TOLERANCE;
    // Spacing the values evenly from start makes the last one exactly ``end``, whatever rounding
    // the requested step carries; on a full circle ``start`` itself is the same angle as ``end``.
    const double spacing = (end - start) / static_cast<double>(intervals);
    for (std::size_t k = axis.periodic ? 1 : 0; k <= intervals; ++k) {
        const double raw = k == intervals ? end : start + static_cast<double>(k) * spacing;
        axis.values.push_back(normalized(raw));
    }
    return axis;
}

ScanGrid build_scan_grid(const ScanOptions& options) {
    options.Validate();
    std::vector<ScanAxis> axes;
    for (unsigned k = 0; k < options.GetNumDihedrals(); ++k) {
        axes.push_back(
            build_scan_axis(options.GetRangeStart(), options.GetRangeEnd(), options.GetStep()));
    }
    return ScanGrid(std::move(axes));
}

}  // namespace detail
}  // namespace OEVIBE
