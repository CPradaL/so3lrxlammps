#!/usr/bin/env python3
"""Generate water box LAMMPS data files at various sizes for scaling tests.

Places water molecules on a cubic grid at ~1.0 g/cm³ density with random
orientations. Outputs LAMMPS atom_style atomic data files with 4 atom types
matching the CHON model (C=1, H=2, O=3, N=4; water uses only types 2 and 3).

Usage:
    python generate_water_boxes.py --output-dir data/
"""

import argparse
import os

import numpy as np

# Element → LAMMPS type mapping (must match pair_coeff * * C H O N)
TYPE_TO_MASS = {1: 12.011, 2: 1.008, 3: 15.999, 4: 14.007}
O_TYPE = 3
H_TYPE = 2

# Water geometry: O-H = 0.9572 A, H-O-H = 104.52°
OH_BOND = 0.9572
HOH_ANGLE = np.radians(104.52)

# Water molecule template (O at origin)
WATER_TEMPLATE = np.array([
    [0.0, 0.0, 0.0],  # O
    [OH_BOND * np.sin(HOH_ANGLE / 2), OH_BOND * np.cos(HOH_ANGLE / 2), 0.0],  # H1
    [-OH_BOND * np.sin(HOH_ANGLE / 2), OH_BOND * np.cos(HOH_ANGLE / 2), 0.0],  # H2
])

# Target sizes: (n_atoms, n_molecules)
SIZES = [
    # Original sizes (strong scaling round 1)
    (6_000, 2_000),
    (12_000, 4_000),
    (24_000, 8_000),
    (48_000, 16_000),
    (96_000, 32_000),
    (192_000, 64_000),
    (384_000, 128_000),
    # 10K atoms/GPU weak scaling (up to 256 GPUs)
    # Max safe: ~12K local atoms/GPU on A100-40GB with SO3LR
    (10_000, 3_334),
    (20_000, 6_667),
    (40_000, 13_334),
    (80_000, 26_667),
    (160_000, 53_334),
    (320_000, 106_667),
    (640_000, 213_334),
    (1_280_000, 426_667),
    (2_560_000, 853_334),
]

# Water density ~ 1.0 g/cm³
# M_water = 18.015 g/mol, N_A = 6.022e23
# rho = n_mol * 18.015 / (N_A * V_cm3)
# V_A3 = n_mol * 18.015 / (6.022e23 * 1.0) * 1e24
# V_A3 = n_mol * 29.93
VOL_PER_MOL = 29.93  # A³ per water molecule at 1 g/cm³


def random_rotation_matrix(rng):
    """Generate a random 3D rotation matrix."""
    # Random quaternion method
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    # Quaternion to rotation matrix
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def generate_water_box(n_molecules, rng):
    """Generate a water box with n_molecules water molecules on a cubic grid."""
    vol = n_molecules * VOL_PER_MOL
    box_side = vol ** (1.0 / 3.0)

    # Grid: find smallest n_grid such that n_grid^3 >= n_molecules
    n_grid = int(np.ceil(n_molecules ** (1.0 / 3.0)))
    spacing = box_side / n_grid

    print(f"  {n_molecules} molecules, box = {box_side:.2f} A, "
          f"grid = {n_grid}³ = {n_grid**3}, spacing = {spacing:.2f} A")

    positions = []
    types = []
    count = 0

    for ix in range(n_grid):
        for iy in range(n_grid):
            for iz in range(n_grid):
                if count >= n_molecules:
                    break
                # Grid center for this molecule
                center = np.array([
                    (ix + 0.5) * spacing,
                    (iy + 0.5) * spacing,
                    (iz + 0.5) * spacing,
                ])
                # Random rotation
                R = random_rotation_matrix(rng)
                mol = (R @ WATER_TEMPLATE.T).T + center

                positions.append(mol[0])  # O
                types.append(O_TYPE)
                positions.append(mol[1])  # H
                types.append(H_TYPE)
                positions.append(mol[2])  # H
                types.append(H_TYPE)
                count += 1
            if count >= n_molecules:
                break
        if count >= n_molecules:
            break

    positions = np.array(positions)
    types = np.array(types)

    # Wrap into box (should already be inside, but just in case)
    positions = positions % box_side

    return positions, types, box_side


def write_data_file(positions, types, box_side, filename):
    """Write LAMMPS data file (atom_style atomic)."""
    natoms = len(positions)
    with open(filename, "w") as f:
        f.write(f"# LAMMPS data file: water box ({natoms} atoms)\n\n")
        f.write(f"{natoms} atoms\n")
        f.write(f"4 atom types\n\n")
        f.write(f"0.00000000 {box_side:.8f} xlo xhi\n")
        f.write(f"0.00000000 {box_side:.8f} ylo yhi\n")
        f.write(f"0.00000000 {box_side:.8f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t, m in sorted(TYPE_TO_MASS.items()):
            f.write(f"{t} {m}\n")
        f.write("\nAtoms\n\n")
        for i in range(natoms):
            f.write(
                f"{i + 1} {types[i]} "
                f"{positions[i][0]:.8f} {positions[i][1]:.8f} {positions[i][2]:.8f}\n"
            )
    print(f"  Wrote {filename} ({natoms} atoms, box = {box_side:.2f} A)")


def main():
    parser = argparse.ArgumentParser(description="Generate water box LAMMPS data files")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip generation if data file already exists")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="Atom counts to generate (default: the built-in list). "
                             "Rounded down to a whole number of water molecules.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sizes = SIZES if args.sizes is None else [(n, n // 3) for n in args.sizes]
    for n_atoms, n_molecules in sizes:
        filename = os.path.join(args.output_dir, f"water_{n_atoms}.data")
        if args.skip_existing and os.path.exists(filename):
            print(f"\nSkipping {filename} (already exists)")
            continue
        print(f"\nGenerating water box with {n_atoms} atoms...")
        positions, types, box_side = generate_water_box(n_molecules, rng)
        write_data_file(positions, types, box_side, filename)

    print("\nDone!")


if __name__ == "__main__":
    main()
