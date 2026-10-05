# SO3LR native source

This directory is the single numerical source for both the native built-in
LAMMPS package and the external plugin. Its numerical implementation is frozen
from optimization dev_12, commit
`99674789160d0bda7c92439fe76cbcb6ecf42580`.

The historical development README from the archived commit is retained as
`README.DEVELOPMENT_HISTORY.md`. Use the helpers in the top-level release
directory; do not maintain separate numerical copies for the two installation
routes.

External plugin build:

```bash
../../tools/build_external_plugin.sh /path/to/lammps /tmp/so3lr-build /path/to/install
```

See the top-level `README.md` for runtime requirements and the LAMMPS input
syntax. The exporter directory is required only to convert a supported
framework checkpoint; it is not used during native MD.
