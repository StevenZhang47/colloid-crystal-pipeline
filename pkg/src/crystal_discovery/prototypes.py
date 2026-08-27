"""Which anchor slugs name the same structure.

The library holds several CIF settings of the same prototype, and a few files whose name
does not match the prototype they describe. Collapsing them is not cosmetic: retrieval's
runner-up margin compares the best match against the best DIFFERENT structure, and two
settings of one prototype sit at distance ~0 from each other. Left uncollapsed they would
fail every candidate for a reason that has nothing to do with the candidate.

The equivalences are not derivable from the filenames, so they are recorded here as data.
Each was established with a role-relabelled StructureMatcher (minority species -> A,
majority -> B), which fits li3fe to ni3sn_D019 at rms 0.045 while correctly REFUSING to
merge the negative controls -- Cu3Au/ReO3, Ni3Sn/I3P, CsCl/NaTl, Laves C14/C36. An earlier
attempt with pymatgen's FrameworkComparator merged CsCl with Cu3Au and had to be discarded.
"""
from __future__ import annotations

# slug (lowercased directory name) -> canonical prototype
ALIASES: dict[str, str] = {
    "cscl": "CsCl",
    "cscl_b2_anchor": "CsCl",
    "th3p4__8edd1f72": "Th3P4",
    "th3p4__72406369": "Th3P4",
    "th3p4_mp-1347_primitive": "Th3P4",
    "th3p4_anchor": "Th3P4",
    "cu3au": "Cu3Au",
    "cu3au_l12": "Cu3Au",
    "cu3au_mp-2258_conventional_standard": "Cu3Au",
    "ni3sn_d019": "Ni3Sn",
    "li3fe": "Ni3Sn",                     # D0_19; the filename does not say so
    "zangenite__471e7cda": "zangenite",
    "zangenite__98bb84a4": "zangenite",
    "cr3si_a15": "Cr3Si_A15",
    "rocksalt_b1": "rocksalt_B1",
    "fluorite_c1": "fluorite",
    "antifluorite_c1b": "fluorite",       # identical once species roles are relabelled
}


def prototype_of(name: str) -> str:
    """Canonical prototype for an anchor slug. Unknown slugs are their own prototype."""
    base = name[5:] if name.startswith("orig_") else name
    return ALIASES.get(base.lower(), base)
