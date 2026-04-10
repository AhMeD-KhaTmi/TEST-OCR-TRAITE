"""Unit tests for verify_phase1.py helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the helper directly from the verify script
import importlib.util, types

_spec = importlib.util.spec_from_file_location(
    "verify_phase1",
    Path(__file__).parent.parent / "verify_phase1.py",
)
# We don't actually execute the script; we just want the _make_doc_id function.
# Extract it by parsing the source and exec-ing only the function definition.
_source = (Path(__file__).parent.parent / "verify_phase1.py").read_text()
_fn_start = _source.index("def _make_doc_id")
_fn_end   = _source.index("\nfor img_path", _fn_start)
_fn_code  = _source[_fn_start:_fn_end]

_module = types.ModuleType("verify_helpers")
_module.Path = Path
exec(compile(_fn_code, "verify_phase1.py", "exec"), _module.__dict__)
_make_doc_id = _module._make_doc_id


class TestMakeDocId:
    def test_batch1_produces_1_prefix(self):
        p = Path("trabelsi.mohamedali_03-04-2026_15-49-11_1_page-0001.jpg")
        assert _make_doc_id(p) == "1_0001"

    def test_batch4_page20(self):
        p = Path("trabelsi.mohamedali_03-04-2026_15-49-11_4_page-0020.jpg")
        assert _make_doc_id(p) == "4_0020"

    def test_pages_to_jpg_pattern(self):
        p = Path("trabelsi.mohamedali_03-04-2026_15-49-11_pages-to-jpg-0001.jpg")
        assert _make_doc_id(p) == "ptj_0001"

    def test_pages_to_jpg_page20(self):
        p = Path("trabelsi.mohamedali_03-04-2026_15-49-11_pages-to-jpg-0020.jpg")
        assert _make_doc_id(p) == "ptj_0020"

    def test_all_five_batches_produce_unique_ids(self):
        pages = [f"{i:04d}" for i in range(1, 21)]
        ids: list[str] = []
        for batch in ("1", "2", "3", "4"):
            for page in pages:
                p = Path(f"trabelsi.mohamedali_03-04-2026_15-49-11_{batch}_page-{page}.jpg")
                ids.append(_make_doc_id(p))
        for page in pages:
            p = Path(f"trabelsi.mohamedali_03-04-2026_15-49-11_pages-to-jpg-{page}.jpg")
            ids.append(_make_doc_id(p))
        assert len(ids) == len(set(ids)), "Duplicate doc IDs detected"

    def test_batch_ids_differ_from_ptj_ids(self):
        p1 = Path("trabelsi.mohamedali_03-04-2026_15-49-11_1_page-0001.jpg")
        p2 = Path("trabelsi.mohamedali_03-04-2026_15-49-11_pages-to-jpg-0001.jpg")
        assert _make_doc_id(p1) != _make_doc_id(p2)
