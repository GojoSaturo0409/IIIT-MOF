# mof_voxelizer/chemistry.py
import logging
import numpy as np
from typing import List, Tuple, Union
from pymatgen.core import Structure, Element

def build_supercell(struct: Structure, Lmin: float) -> Structure:
    """Expands the structure so that all lattice vectors are at least Lmin."""
    a, b, c = struct.lattice.abc
    na = max(1, int(np.ceil(Lmin / a)))
    nb = max(1, int(np.ceil(Lmin / b)))
    nc = max(1, int(np.ceil(Lmin / c)))
    if (na, nb, nc) != (1, 1, 1):
        logging.debug("Expanding supercell: %s -> (%d,%d,%d)", struct.formula, na, nb, nc)
        return struct * (na, nb, nc)
    return struct

def extract_site_charge(site) -> float:
    """Safely attempts to extract partial charge from a Pymatgen Site."""
    props = getattr(site, "properties", {}) or {}
    keys = (
        "partial_charge", "partial_charges", "charge", "q",
        "_atom_site_partial_charge", "_atom_site_charges",
    )
    # Check site properties
    for k in keys:
        if k in props and props[k] is not None:
            try:
                val = props[k]
                return float(val[0]) if isinstance(val, (list, tuple)) else float(val)
            except Exception:
                pass
    
    # Check specie properties
    try:
        sp = site.specie
        if hasattr(sp, "properties") and sp.properties:
            for k in ("partial_charge", "charge", "q"):
                if k in sp.properties:
                    return float(sp.properties[k])
    except Exception:
        pass
    return 0.0

def _get_atomic_number(sp) -> int:
    Z = getattr(sp, "Z", getattr(sp, "atomic_number", None))
    if Z is None and hasattr(sp, "__str__"):
        try:
            el = Element(getattr(sp, "symbol", str(sp)))
            Z = getattr(el, "Z", getattr(el, "atomic_number", None))
        except Exception:
            pass
    return int(Z) if Z is not None else 0

def get_atom_info(site) -> Tuple[str, int, float]:
    """Returns (Symbol, AtomicNumber, Occupancy)."""
    try:
        # Handle disordered structures
        if hasattr(site, "species") and isinstance(site.species, dict):
            items = list(site.species.items())
            if not items:
                raise ValueError("No species")
            # Take majority species
            spp_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            sp, occ = spp_sorted[0]
        else:
            sp = site.specie
            occ = 1.0
        
        sym = getattr(sp, "symbol", None) or str(sp)
        Z = _get_atomic_number(sp)
        return sym, Z, float(occ)
    except Exception:
        return "X", 0, 1.0
