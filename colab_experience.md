# PaulG-LM Colab Run — Experience & Problems

## Date
September 19, 2026

## Objective
Run the full PaulG-LM training pipeline on Google Colab T4 GPU using Colab CLI.
Pipeline: Stage 1 CPT → Stage 2 ONLY → PaulG-LM Stage 2 → Merge → Eval → Judge Board.

## What Was Achieved
- All project scripts, configs, and docs prepared and committed
- Colab CLI v0.7.1 installed and authenticated
- VM `paulglim` provisioned (T4 GPU, Standard shape) — session lost due to idle timeout
- Data tarball uploaded, deps installed (uv), all 12 required files verified on VM
- Package script bug fixed (exclude pattern)
- Stage 1 data loading works (49,178 train blocks, 5,690 val blocks)
- Fixed 5 Python bugs in training scripts (see below)

## What Was Not Achieved
- No training completed. Pipeline fails at Stage 1 due to API incompatibilities.
- No models trained, no eval, no judge board.

## Problems Encountered

### Colab CLI / Workflow Problems

### 1. `--keep` flag does not work with `colab new`
Attempted: `colab new -s paulglim --gpu T4 --keep`
Error: `No such option: --keep`
The `--keep` flag only exists for `colab run`, not `colab new`. Sessions stay alive by default until stopped, so `--keep` is unnecessary with `colab new`.

### 2. Package script exclude pattern blocked required files
The tar command used `--exclude='data/raw/*'` which also blocked explicit includes of `data/raw/fewshot_examples.jsonl`. In tar, exclude patterns take precedence over explicit file listings.
Fix: Changed to exclude specific large files individually instead of the entire `data/raw/` directory.

### 3. `colab exec` does not accept shell commands directly
Attempted: `colab exec -s paulglim 'cd /content && tar xzf ...'`
Error: `Got unexpected extra argument(s)`
`colab exec` either takes `-f` (local file) or reads from stdin. Shell command strings are not accepted as positional arguments.
Workaround: Created `setup_colab.py` and used `colab exec -s paulglim -f /tmp/setup_colab.py`.

### 4. `colab exec -f` requires LOCAL file paths
Attempted: `colab exec -s paulglim -f /content/setup_colab.py` (remote path)
Error: `FileNotFoundError: [Errno 2] No such file or directory: '/content/setup_colab.py'`
The `-f` flag reads a LOCAL file from your machine and sends it to the VM. It does not read files already on the VM.
Fix: Used `/tmp/setup_colab.py` (local path) instead of `/content/setup_colab.py` (remote path).

### 5. `run_on_colab.py` used `__file__` for base directory
When run via `colab exec -f`, the Python code runs inline in the kernel — `__file__` is undefined. The original code used `os.path.abspath(__file__)` which would fail.
Fix: Changed to check `os.path.exists("/content")` first, falling back to `__file__` for local execution.

### 6. Setup script lost between sessions
The `/tmp/setup_colab.py` file was lost when the VM session was terminated. Since `/tmp` is on the local WSL machine, it should have persisted — but Colab CLI may have cleaned up temp files during session transitions.
Fix: Recreated the file each time. Should write it to a persistent location (e.g., project directory).

### 7. VM session was lost (404/401)
After the setup step completed successfully, the VM session `paulglim` was lost (404/401 error). The Colab VM expired or was terminated by Google before the pipeline could be executed. The `colab stop` command also failed because the session no longer existed.
No root cause could be determined. Free-tier Colab VMs have ~90-minute idle timeouts.

### Python Script Bugs (fixed)

### 8. `torchao` version incompatibility with PEFT
Error: `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`
PEFT calls `is_torchao_available()` which fails with old torchao.
Fix: `colab exec -s paulglim -f /tmp/fix_torchao.py` → `pip install --upgrade torchao` (0.10.0 → 0.18.0)

### 9. `Path` not imported in train_stage1.py
Error: `NameError: name 'Path' is not defined` at line 104
`Path` used but not imported (other scripts had the import).
Fix: Added `from pathlib import Path` to imports in `train_stage1.py`

### 10. `LOGGING_STEPS` not defined in train_stage1.py
Error: `NameError: name 'LOGGING_STEPS' is not defined` at line 126
Other stage2 scripts define `LOGGING_STEPS = 10` but Stage 1 was missing it.
Fix: Added `LOGGING_STEPS = 10` to constants in `train_stage1.py`

### 11. `evaluation_strategy` renamed to `eval_strategy` in transformers v4.46+
Error: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
New transformers API uses `eval_strategy` instead of `evaluation_strategy`.
Fix: Replaced `evaluation_strategy` with `eval_strategy` in all 3 training scripts

## Workflow Lessons
1. **Always use `colab install` (uv)** for package installation — faster than pip
2. **`colab exec -f` takes LOCAL paths only** — never remote VM paths
3. **Shell commands in `colab exec`** — use `-f` with a script file, or pipe via stdin
4. **Tar exclude patterns override explicit includes** — list specific files to exclude instead of directory-level patterns
5. **Colab sessions are ephemeral** — download all results before stopping
6. **`colab exec` has 10s poll timeout** — scripts should produce output every ~5s
7. **`--keep` is for `colab run` only** — `colab new` sessions stay alive by default
8. **Check installed package versions** — torchao, transformers, PEFT may have conflicts
9. **Test training scripts individually** before running full pipeline
10. **New transformers API: `eval_strategy` not `evaluation_strategy`**

## Files Created
- `src/training/run_on_colab.py` — Full pipeline runner (6 steps, heartbeat, dependency check)
- `src/training/setup_colab.py` — Extract tarball + verify files on VM (local: `/tmp/setup_colab.py`)
- `src/package_for_colab.sh` — Packages data+scripts+config + prints run instructions
