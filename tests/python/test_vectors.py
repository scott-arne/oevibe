"""The wrapped STL containers expose no entry point into undefined behavior.

``std_vector.i`` wraps ``front()``, ``back()`` and ``pop_back()`` with no size check, so on an
empty vector they are undefined behavior rather than an exception. Nothing in OEVibe needs them.
"""

import pytest

from oevibe import oevibe as _proxy

VECTOR_NAMES = ("DoubleVector", "UnsignedVector", "StringVector", "BondOrderVector")
UNGUARDED = ("front", "back", "pop_back", "get_allocator")
MAP_NAMES = ("StringStringMap", "UnsignedDoubleMap")


@pytest.mark.parametrize("vector_name", VECTOR_NAMES)
@pytest.mark.parametrize("method", UNGUARDED)
def test_vectors_do_not_expose_the_unguarded_methods(vector_name, method):
    assert not hasattr(getattr(_proxy, vector_name), method)


def test_double_vector_still_reads_by_index_and_pops_safely():
    vector = _proxy.DoubleVector()
    vector.append(1.5)
    vector.append(2.5)
    assert len(vector) == 2
    assert vector[1] == 2.5
    assert list(vector) == [1.5, 2.5]
    assert vector.pop() == 2.5
    vector.pop()
    with pytest.raises(IndexError):
        vector.pop()


@pytest.mark.parametrize("map_name", MAP_NAMES)
def test_maps_iterate_over_a_key_snapshot(map_name):
    """``__iter__`` must be the snapshot version, not the dead one over the ignored iterator."""
    mapping = getattr(_proxy, map_name)()
    assert not hasattr(mapping, "key_iterator")
    assert list(mapping) == []
