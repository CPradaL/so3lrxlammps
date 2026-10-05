# Converting a checkpoint to `.so3lr`

LAMMPS reads only native `.so3lr` files. The SO3LR model is included as
`models/so3lr.so3lr`; conversion is needed only for other or retrained
checkpoints. It is done once, in a Python environment; Python, JAX and PyTorch
are conversion-time dependencies only.

## JAX/flax checkpoints: `export_flax_so3lr.py`

Reads the flax parameters and training configuration directly and writes the
complete architecture record, so the runtime can check at load time that it
supports the model.

```bash
cd tools/checkpoint_conversion
# the so3krates_torch sources are needed for their tensor-name table only
export PYTHONPATH=/path/to/So3krates-torch/src:$PYTHONPATH

python export_flax_so3lr.py --model so3lr --output so3lr.so3lr
python export_flax_so3lr.py --model /path/to/workdir --output my-model.so3lr
```

`--model` is the name of a checkpoint bundled with the `so3lr` Python package
(SO3LR itself is `so3lr`) or a training work directory. Options:
`--lr-cutoff` (default 12 Å), `--lr-damping`, `--theory-level`,
`--verify-against FILE` (compare every tensor with an existing `.so3lr`). Check
the result:

```bash
python so3lr_native_format.py my-model.so3lr --manifest my-model.manifest.json
so3lr_capability_probe my-model.so3lr      # can this build run it?
```

## PyTorch checkpoints: `convert_so3lr_checkpoint.py`

For `so3krates_torch` checkpoints (`SO3LR`, `MultiHeadSO3LR`, or a
`LAMMPS_MLIAP_SO3` wrapper):

```bash
export PYTHONPATH=/path/to/So3krates-torch/src:$PYTHONPATH
python convert_so3lr_checkpoint.py \
  --model /path/to/checkpoint.model \
  --output /path/to/model.so3lr \
  --source-commit COMMIT_OF_SO3KRATES_TORCH \
  --long-range 12.0 \
  --max-native-z 99 \
  --summary /path/to/model.conversion.json
```

On MeluXina, `examples/meluxina/submit_convert_checkpoint.sh` runs this inside
a container that provides PyTorch and `so3krates_torch`.

## Checking a converted model

Run a single point against the JAX reference:

```bash
python tests/jaxref/so3lr_reference.py --model so3lr --data system.data \
    --types C H O N --output ref.json [--periodic] [--charge Q --multiplicity M]
lmp -var DATA_FILE system.data -var MODEL_FILE models/so3lr.so3lr \
    -var ELEMENTS "C H O N" -var BOUNDARY "s s s" -var DUMP f.dump \
    -in tests/jaxref/in.single_point -log sp.log
python tests/jaxref/compare_native.py ref.json f.dump sp.log
```

`tests/jaxref/refs/` already holds references for SO3LR on the test systems.
For charged or open-shell systems, add `--float64-params` to the reference
(see VALIDATION.md for why). Record SHA-256 checksums of the source checkpoint,
the converter and the output; conversion is deterministic.
