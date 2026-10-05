# Models

| file | model | short / long-range cutoff | comfortable atoms per A100-40GB |
|---|---|---|---|
| `so3lr.so3lr` | SO3LR | 4.5 / 12 Å | ~24,000 |

A `.so3lr` file holds the learned tensors, physical reference tables, an
architecture record, provenance metadata and checksums. The pair styles need
nothing else; no Python runs during MD. `so3lr.so3lr` was exported from the
JAX checkpoint of the `so3lr` package with
`tools/checkpoint_conversion/export_flax_so3lr.py --model so3lr`. It is the
file every result in [../VALIDATION.md](../VALIDATION.md) was obtained with.
It supports elements Z = 1–99 and carries the charge and spin embeddings used
by the pair styles' optional total-charge and multiplicity arguments.

To convert another checkpoint, see
[../docs/MODEL_CONVERSION.md](../docs/MODEL_CONVERSION.md). A model is checked
against the build's declared capabilities when it is loaded; an unsupported
architecture is refused with a list of the offending features.
`so3lr_capability_probe MODEL` (built with the native library) answers the
same question without LAMMPS.

Other `.so3lr` files placed here are ignored by git: redistributing weights
needs the SO3LR authors' approval ([../LICENSE_STATUS.md](../LICENSE_STATUS.md)).
