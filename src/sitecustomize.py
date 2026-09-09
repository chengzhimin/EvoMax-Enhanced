"""Compatibility aliases for the pinned fair-esm dependency set."""

try:
    import biotite.structure as structure

    if not hasattr(structure, "filter_backbone"):
        structure.filter_backbone = structure.filter_peptide_backbone
except ImportError:
    # The hook must not prevent environments without Biotite from importing.
    pass
