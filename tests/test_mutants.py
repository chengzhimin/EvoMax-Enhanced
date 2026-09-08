from evomax_enhanced.core.pipeline import enumerate_single_mutants


def test_enumeration_has_19_substitutions_per_residue():
    mutants = enumerate_single_mutants("AC")
    assert len(mutants) == 38
    assert all(wt != mut for _, wt, mut in mutants)
    assert {p for p, _, _ in mutants} == {0, 1}
