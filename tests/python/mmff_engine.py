"""A test-only MMFF94 engine written in Python (spec 8.7).

It re-exposes OpenEye's MMFF through the Python toolkit, so optimizing with it must reproduce the
C++ ``oeff`` engine exactly. That makes it the end-to-end check of the director path, the spec 3.1
units, and the dense index mapping.
"""

import numpy as np
from openeye import oechem, oeff

import oevibe

ENGINE_NAME = "mmff-python"


class MMFFSession(oevibe.PythonEngineSession):
    """One MMFF potential set up on its own copy of the snapshot."""

    def __init__(self, molecule, mmff94s):
        super().__init__()
        # The potential keeps pointers into the molecule it was set up on, so the session owns it.
        self.molecule = oechem.OEMol(molecule)
        self.potential = oeff.OEMMFF(mmff94s)
        # sweep=False: a sweep may renumber atoms, which would break the dense index mapping.
        if not self.potential.PrepMol(self.molecule, False, True):
            raise oevibe.UnsupportedError("MMFF preparation failed")
        if not self.potential.Setup(self.molecule):
            raise oevibe.UnsupportedError(
                "MMFF setup failed; parameters may be missing"
            )
        self.num_var = self.potential.NumVar()
        if self.num_var != 3 * self.molecule.NumAtoms():
            raise oevibe.UnsupportedError(
                f"MMFF expects {self.num_var} variables, not {3 * self.molecule.NumAtoms()}"
            )

    def evaluate(self, coords, request):
        flat = oechem.OEDoubleArray(coords.reshape(-1))
        if not request.gradient:
            return oevibe.Evaluation(energy=self.potential(flat))
        gradient = oechem.OEDoubleArray(self.num_var)
        energy = self.potential(flat, gradient)
        values = np.fromiter(
            (gradient[index] for index in range(self.num_var)),
            dtype=np.float64,
            count=self.num_var,
        )
        return oevibe.Evaluation(energy=energy, gradient=values)


class MMFFEngine(oevibe.PythonEngine):
    """MMFF94 and MMFF94s in the gas phase, through the OpenEye Python toolkit."""

    def __init__(self, name=ENGINE_NAME):
        super().__init__()
        self._name = name

    def GetName(self):
        return self._name

    def GetVersion(self):
        return oeff.OEFFGetRelease()

    def GetMethods(self):
        methods = []
        for method in ("mmff94", "mmff94s"):
            capabilities = oevibe.MethodCapabilities()
            capabilities.method = oevibe.Method(method)
            capabilities.analytic_gradient = True
            capabilities.charge_from_topology = True
            capabilities.force_field = True
            methods.append(capabilities)
        return methods

    def CreateSession(self, system, settings):
        return MMFFSession(system.molecule, settings.method.GetName() == "mmff94s")


def make_engine():
    """:returns: A new engine; the zero-argument callable an ``oevibe.engines`` entry point needs."""
    return MMFFEngine()
