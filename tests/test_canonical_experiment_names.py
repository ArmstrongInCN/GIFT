"""Keep public experiment identities consistent with the manuscript."""
from pathlib import Path
import re

from experiments.paper_nomenclature import PAPER_EXPERIMENT_ORDER, resolve_experiment_id

ROOT = Path(__file__).resolve().parents[1]


def test_all_entrypoints_use_the_six_canonical_experiment_ids():
    assert PAPER_EXPERIMENT_ORDER == ('W1', 'W2', 'W3', 'S1', 'S2', 'S3')
    for identifier in PAPER_EXPERIMENT_ORDER:
        assert resolve_experiment_id(identifier) == identifier


def test_no_obsolete_task_paths_or_identifiers_in_active_python_code():
    obsolete = [chr(77) + str(index) for index in range(1, 4)] + [chr(83) + str(4)]
    tokens = '|'.join(re.escape(value) for value in (*obsolete, *(x.lower() for x in obsolete)))
    pattern = re.compile(r'(?<![A-Za-z0-9])(?:' + tokens + r')(?![A-Za-z0-9])')
    failures = []
    for directory in ('src', 'training', 'scripts', 'experiments', 'tests'):
        for path in (ROOT / directory).rglob('*.py'):
            relative = path.relative_to(ROOT).as_posix()
            if pattern.search(relative) or pattern.search(path.read_text(encoding='utf-8-sig')):
                failures.append(relative)
    assert not failures, failures
