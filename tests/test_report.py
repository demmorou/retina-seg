from src.api.report import MANDATORY_DISCLAIMER, build_report


def test_build_report_no_lesions_states_nothing_found():
    report = build_report("Sem sinais de retinopatia diabética", {"MA": 0, "HE": 0, "EX": 0, "SE": 0})
    assert "Nenhuma lesão" in report
    assert MANDATORY_DISCLAIMER in report


def test_build_report_lists_counts_by_type():
    report = build_report("Retinopatia diabética não proliferativa moderada", {"MA": 5, "HE": 2, "EX": 0, "SE": 0})
    assert "5 microaneurismas" in report
    assert "2 hemorragias" in report
    assert "exsudatos" not in report  # EX/SE zerados não aparecem na listagem
    assert MANDATORY_DISCLAIMER in report


def test_build_report_always_includes_disclaimer():
    for summary in ({"MA": 0}, {"MA": 100, "HE": 50}):
        assert MANDATORY_DISCLAIMER in build_report("qualquer grau", summary)
