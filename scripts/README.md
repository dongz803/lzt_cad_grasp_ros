# Script directory map

The scripts are grouped by responsibility. Run commands from the repository root
unless a document says otherwise.

- `pipeline/`: end-to-end entry points and FoundPose stage orchestration.
- `perception/`: pose conversion, gating, mask/pose refinement, and reliability.
- `grasp/`: CAD grasp generation, frame transformation, and robust ranking.
- `evaluation/`: BOP evaluation, metrics, visualization, baseline freezing, and ablations.
- `datasets/`: T-LESS/ROBI conversion, preparation, rendering, and validation.
- `bootstrap/`: third-party source download helpers.
- `analysis/`: offline uncertainty and pose-mode analysis.
- `_bootstrap.py`: shared local-package path setup used by scripts in the subdirectories.

The directory grouping is organizational only; it does not change the algorithms or
their inputs and outputs.
