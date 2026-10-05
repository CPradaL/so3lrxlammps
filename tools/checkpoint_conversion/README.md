# Checkpoint conversion

| script | input | use |
|---|---|---|
| `export_flax_so3lr.py` | JAX/flax checkpoint (`so3lr` package) | SO3LR to `.so3lr` |
| `convert_so3lr_checkpoint.py` | `so3krates_torch` checkpoint | SO3LR from PyTorch to `.so3lr` |
| `so3lr_native_format.py` | `.so3lr` file | read, verify and dump the manifest of a native model |

Usage and checks: [../../docs/MODEL_CONVERSION.md](../../docs/MODEL_CONVERSION.md).
