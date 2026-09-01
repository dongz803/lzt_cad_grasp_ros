import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config" / "foundpose_smoke"


def load_options(filename: str, key: str) -> dict:
    return json.loads((CONFIG_ROOT / filename).read_text(encoding="utf-8"))[key]


def test_foundpose_smoke_configs_use_one_consistent_tless_object() -> None:
    templates = load_options("tless_templates.json", "gen_templates_opts")
    representation = load_options("tless_repre.json", "gen_repre_opts")
    inference = load_options("tless_infer.json", "infer_opts")

    for options in (templates, representation, inference):
        assert options["object_dataset"] == "tless"
        assert options["object_lids"] == [1]

    assert templates["version"] == representation["templates_version"]
    assert representation["version"] == inference["repre_version"]
    assert representation["extractor_name"] == inference["extractor_name"]


def test_foundpose_helpers_are_valid_python() -> None:
    for filename in ("prepare_tless_smoke.py", "run_foundpose.py"):
        source = (PROJECT_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        ast.parse(source, filename=filename)
