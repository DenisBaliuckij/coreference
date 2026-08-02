import sys
from pathlib import Path

from pipeline.sys_path_setup import add_text_corpuses_processing_to_path


def test_default_path_points_at_sibling_dags_dir():
    result = add_text_corpuses_processing_to_path()
    assert result == Path(__file__).resolve().parents[2] / "text-corpuses-processing" / "dags"
    assert str(result) in sys.path


def test_env_var_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_CORPUSES_PROCESSING_DAGS", str(tmp_path))
    result = add_text_corpuses_processing_to_path()
    assert result == tmp_path
    assert str(tmp_path) in sys.path


def test_idempotent_does_not_duplicate_sys_path_entry():
    sys.path[:] = [p for p in sys.path if "text-corpuses-processing" not in p]
    add_text_corpuses_processing_to_path()
    count_after_first = sum(1 for p in sys.path if "text-corpuses-processing" in p)
    add_text_corpuses_processing_to_path()
    count_after_second = sum(1 for p in sys.path if "text-corpuses-processing" in p)
    assert count_after_first == 1
    assert count_after_second == 1
