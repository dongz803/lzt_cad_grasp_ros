import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "lzt_perception" / "config" / "foundpose_smoke"


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
    script_paths = (
        "scripts/datasets/prepare_tless_smoke.py",
        "scripts/pipeline/run_foundpose.py",
    )
    for relative_path in script_paths:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        filename = Path(relative_path).name
        ast.parse(source, filename=filename)


def test_foundpose_inference_uses_configured_representation_version() -> None:
    patch_name = "foundpose_repre_version.patch"
    patch = (
        PROJECT_ROOT / "third_party" / "patches" / patch_name
    ).read_text(encoding="utf-8")
    fetch_source = (PROJECT_ROOT / "scripts/bootstrap/fetch_foundpose_raw.py").read_text(
        encoding="utf-8"
    )
    freeze_source = (
        PROJECT_ROOT / "scripts/evaluation/freeze_foundpose_baseline.py"
    ).read_text(encoding="utf-8")

    assert "-            base_repre_dir, opts.version," in patch
    assert "+            base_repre_dir, opts.repre_version," in patch
    assert patch_name in fetch_source
    assert patch_name in freeze_source
