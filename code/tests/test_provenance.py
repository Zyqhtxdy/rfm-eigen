import json
from pathlib import Path

import pytest

from rfmeig import provenance


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, 'DATA_ROOT', tmp_path)
    return tmp_path


def test_resume_preserves_completed_draw_and_rejects_changed_configuration(run_root):
    run = provenance.open_run('test', config={'seed': 8}, run_id='one')
    run.complete('draw0', {'value': 3.25})
    again = provenance.open_run('test', config={'seed': 8}, run_id='one', resume=True)
    assert again.completed('draw0')['value'] == 3.25
    with pytest.raises(RuntimeError, match='configuration'):
        provenance.open_run('test', config={'seed': 9}, run_id='one', resume=True)
    with pytest.raises(FileExistsError):
        again.complete('draw0', {'value': 8.})


def test_sealed_run_rejects_writes_even_through_a_previously_returned_path(run_root):
    run = provenance.open_run('test', config={}, run_id='sealed')
    path = run.path('rows.csv')
    provenance.write_csv(path, [{'value': 2.}])
    run.seal()
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match='sealed'):
        provenance.write_csv(path, [{'value': 7.}])
    with pytest.raises(RuntimeError, match='sealed'):
        run.complete('draw', {'value': 7.})
    with pytest.raises(RuntimeError, match='sealed'):
        run.path('new.json')
    with pytest.raises(RuntimeError, match='sealed'):
        provenance.open_run('test', config={}, run_id='sealed', resume=True)
    assert path.read_bytes() == before
    manifest = json.loads((run.directory / 'output_manifest.json').read_text())
    assert manifest['files']['rows.csv'] == provenance.hash_file(path)


@pytest.mark.parametrize('name', ['../escape', '..', '/absolute', 'C:\\escape', 'a/b'])
def test_run_names_cannot_escape_output_root(run_root, name):
    with pytest.raises(ValueError):
        provenance.open_run('test', config={}, run_id=name)


def test_run_paths_cannot_escape_output_root(run_root):
    run = provenance.open_run('test', config={}, run_id='safe')
    with pytest.raises(ValueError):
        run.path('..', '..', 'escaped.txt')
    assert not (Path(run_root).parent / 'escaped.txt').exists()


def test_loaded_thread_pool_mismatch_is_detected(monkeypatch):
    monkeypatch.setattr('threadpoolctl.threadpool_info', lambda: [{'prefix': 'blas', 'num_threads': 7}])
    with pytest.raises(RuntimeError, match='loaded numerical thread pools'):
        provenance.require_threads(1)


def test_experiment2_resumes_after_summary_write_without_repeating_solves(run_root, monkeypatch):
    from rfmeig.experiments import exp2_graded_ball as exp2

    monkeypatch.setattr(exp2, 'DEFAULT_FEM', (1,))
    monkeypatch.setattr(exp2, 'DEFAULT_PIELM', ((1, 1),))
    monkeypatch.setattr(exp2, 'DEFAULT_RFM', ((16, 4),))
    calls = {'scale': 0, 'fem': 0, 'pielm': 0, 'rfm': 0, 'quadrature': 0}

    def scale(*args):
        calls['scale'] += 1
        return 1., .1

    def solve(key, method):
        def result(*args):
            calls[key] += 1
            return {'method': method, 'setting': key, 'degrees_of_freedom': 16, 'admissible': 16,
                    'median_retained': 16, 'median_seconds': 1., 'median_error': .001,
                    'min_error': .001, 'max_error': .001, 'clusters': [], 'draws': 1}
        return result

    def sensitivity(*args):
        calls['quadrature'] += 1
        return [{'level': 4, 'error': .001}]

    monkeypatch.setattr(exp2, 'select_scale_blind', scale)
    monkeypatch.setattr(exp2, 'fem_row', solve('fem', 'P2 FEM'))
    monkeypatch.setattr(exp2, 'pielm_row', solve('pielm', 'Eig-PIELM'))
    monkeypatch.setattr(exp2, 'rfm_row', solve('rfm', 'RFM'))
    monkeypatch.setattr(exp2, 'quadrature_sensitivity', sensitivity)
    write_csv = provenance.write_csv

    def interrupted(*args):
        raise RuntimeError('simulated interruption before CSV write')

    monkeypatch.setattr(provenance, 'write_csv', interrupted)
    argv = ['--run-id', 'resume-exp2', '--threads', '1', '--count', '1', '--draws', '1', '--repeats', '1']
    with pytest.raises(RuntimeError, match='simulated interruption'):
        exp2.main(argv)
    monkeypatch.setattr(provenance, 'write_csv', write_csv)
    exp2.main(argv + ['--resume'])
    assert calls == dict.fromkeys(calls, 1)
    assert (run_root / 'experiment2/resume-exp2/output_manifest.json').is_file()
