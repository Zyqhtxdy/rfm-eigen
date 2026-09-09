import hashlib

import pandas as pd
import pytest

import audit
import reproduce
from paper_assets import chinese_tables
from paper_assets.paths import REPO_DATA
from paper_assets.records import converged_rows, repetitions_per_draw
from rfmeig.experiments.exp2_graded_ball import verdict


def test_rebuild_checks_all_five_tables_without_changing_inputs(tmp_path):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in REPO_DATA.glob('*_table.tex')}
    report = audit.rebuild_tables(tmp_path)
    assert len(report) == 5
    assert all(row['rebuilt'] and row['unchanged_by_rebuild'] for row in report.values())
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in REPO_DATA.glob('*_table.tex')}
    assert before == after


def test_audit_distinguishes_variable_draw_counts_from_timing_repeats():
    counts = audit.check_draw_counts()
    assert counts['experiment5_nonlinear_gpe_table.tex']['recorded_in_run'] == {
        128: 19, 256: 20, 384: 18, 512: 19, 768: 19,
    }
    assert audit.rfm_draw_count_in('N=100; no number of feature draws is stated') is None
    with pytest.raises(ValueError):
        audit.variable_draw_counts_in('sample sizes for N=128,256 are 20, respectively')


def test_false_strings_are_not_treated_as_converged():
    frame = pd.DataFrame({'converged': ['True', 'False', '0', '1'], 'value': [1, 2, 3, 4]})
    assert converged_rows(frame).value.tolist() == [1, 4]
    with pytest.raises(ValueError):
        converged_rows(pd.DataFrame({'converged': [None]}))
    with pytest.raises(ValueError):
        repetitions_per_draw(pd.DataFrame({'features': [128]*3, 'seed': [1, 1, 2]}))


def test_chinese_tables_preserve_numeric_cells_and_current_statistics():
    for name, caption, header in chinese_tables.table_specs():
        english = (REPO_DATA / name).read_text(encoding='utf-8')
        translated = chinese_tables.translate(english, caption, header)
        assert translated.split('\\midrule', 1)[1] == english.split('\\midrule', 1)[1]
        if name.startswith('experiment1'):
            assert '算术均值' in translated and 'Q_{0.9}' not in translated
            assert 'd_{H^1}' in translated
        if name.startswith('experiment5'):
            assert '19, 20, 18, 19, 19' in translated


def test_reproduce_rejects_nonpublic_names():
    with pytest.raises(SystemExit) as error:
        reproduce.main(['_assets'])
    assert error.value.code == 2


def test_absent_baseline_does_not_count_as_a_win():
    result = verdict([{'method': 'RFM', 'setting': 'N=100', 'median_error': 1e-5, 'median_seconds': 1.}])
    assert not result['P2 FEM']['beaten']
    assert not result['Eig-PIELM']['beaten']


def test_matched_accuracy_verdict_uses_the_matched_setting():
    rows = [
        {'method': 'RFM', 'setting': 'N=100', 'median_error': 1e-4, 'median_seconds': 1.},
        {'method': 'RFM', 'setting': 'N=1000', 'median_error': 1e-6, 'median_seconds': 20.},
        {'method': 'P2 FEM', 'setting': 'n=3', 'median_error': 1e-3, 'median_seconds': 10.},
    ]
    result = verdict(rows)['P2 FEM']
    assert result['beaten']
    assert result['matched_setting'] == 'N=100'
    assert result['speed_factor_at_matched_accuracy'] == 10.
