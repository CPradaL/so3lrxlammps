# SO3LR native model format, version 1

## Purpose

The `.so3lr` format transfers a trained SO3LR model from JAX or PyTorch into
the native `libso3lr` runtime. It stores numerical state and an explicit model
contract, not executable Python and not a serialized framework object.

All integer fields and tensor scalars are little-endian. Version 1 initially
supports the exact double-precision SO3LR architecture frozen in Stage 3.0.

## File layout

```text
byte 0     fixed header (128 bytes)
byte 128   canonical UTF-8 JSON manifest
           tensor payload, offsets relative to payload byte 0
```

The fixed header uses this little-endian structure:

```text
8s    magic: SO3LRN1\0
u32   format version: 1
u32   flags: 0
u64   manifest byte count
u64   payload byte count
32B   SHA-256 of exact manifest bytes
32B   SHA-256 of exact payload bytes, including alignment padding
32B   reserved zero bytes
```

Total header size is exactly 128 bytes. The manifest immediately follows the
header; the payload immediately follows the manifest without external padding.

## Manifest

The manifest is canonical JSON: UTF-8, sorted keys and compact separators. It
contains:

- format/schema version and endianness;
- source model name and SHA-256;
- source implementation commit and exporter version;
- model-family and physical/architectural configuration;
- complete module hierarchy and supported module-type registry;
- tensor table with name, role, dtype, shape, aligned payload offset, byte
  count and SHA-256.

Tensor offsets are relative to the first payload byte. Every tensor starts on
a 64-byte boundary. Padding bytes must be zero. Tensor names are unique and
sorted, making exports deterministic.

## Version-1 dtypes

| Manifest dtype | Scalar representation |
|---|---|
| `float64` | IEEE-754 little-endian binary64 |
| `float32` | IEEE-754 little-endian binary32 |
| `int64` | little-endian signed 64-bit integer |
| `int32` | little-endian signed 32-bit integer |
| `uint8` | unsigned byte |
| `bool` | one byte, zero or one |

The current water checkpoint is expected to contain `float64` and `int64`
tensors. Additional dtypes require a format-version update or an explicitly
backward-compatible reader change.

## Integrity and compatibility

A reader must reject:

- wrong magic, header size, version, flags or reserved bytes;
- noncanonical or malformed JSON;
- manifest/payload length or SHA-256 mismatch;
- duplicate names, unsupported dtypes or invalid shapes;
- unaligned, overlapping or out-of-range tensor records;
- nonzero alignment padding;
- per-tensor hash or byte-count mismatch.

Loading a file proves data integrity, not that a runtime implements every
module type. `libso3lr` must compare the manifest module registry and model
family against its compiled capability table before executing the model.
