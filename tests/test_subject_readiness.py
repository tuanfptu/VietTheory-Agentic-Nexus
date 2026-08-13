import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path("scripts/audit_subject_readiness.py")
    spec = importlib.util.spec_from_file_location("audit_subject_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_local_subject_artifacts_pass_readiness_audit() -> None:
    report = _load_script().build_report(Path.cwd())

    assert report["all_subjects_ready"] is True
    assert len(report["subjects"]) == 5
    assert all(subject["ready"] for subject in report["subjects"])
