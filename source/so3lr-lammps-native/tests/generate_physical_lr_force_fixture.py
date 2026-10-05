#!/usr/bin/env python3
"""Generate a PyTorch oracle for native SR energy plus physical SO3LR LR."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from so3krates_torch.blocks.physical_potentials import (
    ALPHAS,
    BOHR,
    C6_COEF,
    FINE_STRUCTURE,
    HARTREE,
)


def discover_root(wrapper: object) -> torch.nn.Module:
    roots: list[torch.nn.Module] = []
    seen = {id(wrapper)}

    def visit(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            roots.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for value in vars(wrapper).values():
        visit(value)
    if len(roots) != 1:
        raise RuntimeError(f"expected one root module, found {len(roots)}")
    return roots[0]


def find_so3lr(root: torch.nn.Module) -> torch.nn.Module:
    for module in root.modules():
        if type(module).__name__ == "SO3LR":
            return module
    raise RuntimeError("checkpoint does not contain SO3LR")


def flat(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().contiguous().reshape(-1).tolist()


def scatter_forces(
    edge_gradients: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    nodes: int,
) -> torch.Tensor:
    forces = torch.zeros((nodes, 3), dtype=edge_gradients.dtype)
    forces.index_add_(0, senders, edge_gradients)
    forces.index_add_(0, receivers, -edge_gradients)
    return forces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions-json", type=Path)
    parser.add_argument("--periodic-box-length", type=float)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    os.environ.pop("SO3_STAGE2_LR_ANALYTIC_FORCE", None)
    wrapper = torch.load(args.model, map_location="cpu", weights_only=False)
    model = find_so3lr(discover_root(wrapper)).eval()

    reference_positions = [
        [0.000000, 0.000000, 0.000000],
        [0.957200, 0.000000, 0.000000],
        [-0.239987, 0.927297, 0.000000],
        [5.650000, 0.420000, 0.180000],
        [6.505000, 0.760000, 0.110000],
        [5.015000, 1.120000, 0.250000],
    ]
    snapshot_box_length: float | None = None
    if args.positions_json is not None:
        payload = json.loads(args.positions_json.read_text(encoding="utf-8"))
        reference_positions = payload["positions"]
        if "box_length" in payload:
            snapshot_box_length = float(payload["box_length"])
        elif "box_lengths" in payload:
            box_lengths = [float(value) for value in payload["box_lengths"]]
            if len(box_lengths) != 3 or max(box_lengths) - min(box_lengths) > 1.0e-10:
                raise RuntimeError("dynamic fixture requires a cubic box")
            snapshot_box_length = box_lengths[0]
        if len(reference_positions) == 18:
            reference_positions = [
                reference_positions[index:index + 3]
                for index in range(0, 18, 3)
            ]
        if len(reference_positions) != 6 or any(
            len(row) != 3 for row in reference_positions
        ):
            raise RuntimeError("dynamic fixture requires six 3D positions")
    positions0 = torch.tensor(reference_positions, dtype=torch.float64)
    box_length = args.periodic_box_length
    if box_length is None:
        box_length = snapshot_box_length
    elif snapshot_box_length is not None and abs(box_length - snapshot_box_length) > 1.0e-10:
        raise RuntimeError("command-line and snapshot box lengths disagree")
    if box_length is not None and (not math.isfinite(box_length) or box_length <= 0.0):
        raise RuntimeError("periodic box length must be positive and finite")

    def edge_vectors(positions: torch.Tensor, receivers: torch.Tensor,
                     senders: torch.Tensor) -> torch.Tensor:
        vectors = positions[receivers] - positions[senders]
        if box_length is not None:
            vectors = vectors - box_length * torch.round(vectors / box_length)
        return vectors

    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.long)
    nodes = int(atomic_numbers.numel())

    sr_s: list[int] = []
    sr_r: list[int] = []
    lr_s: list[int] = []
    lr_r: list[int] = []
    for sender in range(nodes):
        for receiver in range(nodes):
            if sender == receiver:
                continue
            pair_vector = edge_vectors(
                positions0, torch.tensor([receiver]), torch.tensor([sender])
            )[0]
            radius = float(torch.linalg.vector_norm(pair_vector))
            if radius < 4.5:
                sr_s.append(sender)
                sr_r.append(receiver)
            if sender > receiver and radius < 12.0:
                lr_s.append(sender)
                lr_r.append(receiver)
    sr_senders = torch.tensor(sr_s, dtype=torch.long)
    sr_receivers = torch.tensor(sr_r, dtype=torch.long)
    lr_senders = torch.tensor(lr_s, dtype=torch.long)
    lr_receivers = torch.tensor(lr_r, dtype=torch.long)
    sr_edges = int(sr_senders.numel())
    lr_pairs = int(lr_senders.numel())
    if sr_edges == 0 or lr_pairs == 0:
        raise RuntimeError("empty SR/LR water-dimer topology")

    one_hot = torch.nn.functional.one_hot(
        atomic_numbers - 1, num_classes=118
    ).to(torch.float64)
    # SO3LR divides the sum of element/charge/spin embeddings by
    # sqrt(num_embeddings).  Charge and spin contributions vanish
    # for this neutral closed-shell fixture, but the scale remains.
    embedding = (
        model.inv_feature_embedding(one_hot) / model.embedding_scale
    ).detach()
    initial_ev = torch.zeros((nodes, 24), dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)
    total_charge = torch.tensor([0.0], dtype=torch.float64)
    data = {"node_attrs": one_hot}
    half_mask = torch.ones(lr_pairs, dtype=torch.bool)
    cutoff_lr = float(model.r_max_lr)
    electrostatic_sigma = float(model.electrostatic_energy_scale)
    dispersion_damping = float(model.dispersion_energy_cutoff_lr_damping)
    dispersion_scale = float(model.dispersion_energy_scale)

    def evaluate(
        sr_vectors: torch.Tensor, lr_vectors: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        sr_lengths = torch.linalg.vector_norm(sr_vectors, dim=1, keepdim=True)
        sh = model.spherical_harmonics(+sr_vectors / sr_lengths)
        radial = model.radial_embedding(sr_lengths)
        cutoff = model.cutoff_fn(sr_lengths)
        inv = embedding
        ev = initial_ev
        for block in model.euclidean_transformers:
            inv, ev = block(
                inv, ev, radial, sr_senders, sr_receivers, sh, cutoff,
                return_att=False,
            )
        atomic_energy = model.atomic_energy_output_block(
            inv, data, atomic_numbers=atomic_numbers
        ).reshape(-1)
        charges = model.partial_charges_output_block(
            inv_features=inv,
            atomic_numbers=atomic_numbers,
            total_charge=total_charge,
            batch_segments=batch,
            num_graphs=1,
        ).reshape(-1)
        hirshfeld = model.hirshfeld_output_block(
            inv, atomic_numbers
        ).reshape(-1)
        lr_lengths = torch.linalg.vector_norm(lr_vectors, dim=1)
        electrostatic = model.electrostatic_potential(
            charges, lr_senders, lr_receivers, lr_lengths,
            num_nodes=nodes, cutoff_lr=cutoff_lr,
            electrostatic_energy_scale=electrostatic_sigma,
            local_half_mask=half_mask,
        ).reshape(-1)
        dispersion = model.dispersion_potential(
            hirshfeld, atomic_numbers, lr_senders, lr_receivers, lr_lengths,
            num_nodes=nodes, cutoff_lr=cutoff_lr,
            cutoff_lr_damping=dispersion_damping,
            dispersion_energy_scale=dispersion_scale,
            local_half_mask=half_mask,
        ).reshape(-1)
        return (
            atomic_energy, charges, hirshfeld, electrostatic, dispersion,
            sr_lengths, sh, lr_lengths,
        )

    base_sr_vectors = edge_vectors(positions0, sr_receivers, sr_senders)
    base_lr_vectors = edge_vectors(positions0, lr_receivers, lr_senders)
    sr_vectors = base_sr_vectors.detach().clone().requires_grad_(True)
    lr_vectors = base_lr_vectors.detach().clone().requires_grad_(True)
    (
        atomic_energy, charges, hirshfeld, electrostatic, dispersion,
        sr_lengths, sh_vectors, lr_lengths,
    ) = evaluate(sr_vectors, lr_vectors)
    learned_total = atomic_energy.sum()
    electrostatic_total = electrostatic.sum()
    dispersion_total = dispersion.sum()
    physical_lr_total = electrostatic_total + dispersion_total
    total_energy = learned_total + physical_lr_total
    charge_gradient = torch.autograd.grad(
        physical_lr_total, charges, retain_graph=True
    )[0]
    hirshfeld_gradient = torch.autograd.grad(
        physical_lr_total, hirshfeld, retain_graph=True
    )[0]
    lr_radial_gradient = torch.autograd.grad(
        physical_lr_total, lr_lengths, retain_graph=True
    )[0]
    sr_edge_gradient, lr_edge_gradient = torch.autograd.grad(
        total_energy, (sr_vectors, lr_vectors)
    )
    implicit_forces = scatter_forces(
        sr_edge_gradient, sr_senders, sr_receivers, nodes
    )
    direct_lr_forces = scatter_forces(
        lr_edge_gradient, lr_senders, lr_receivers, nodes
    )
    assembled_forces = implicit_forces + direct_lr_forces

    positions = positions0.detach().clone().requires_grad_(True)
    position_outputs = evaluate(
        edge_vectors(positions, sr_receivers, sr_senders),
        edge_vectors(positions, lr_receivers, lr_senders),
    )
    position_total = (
        position_outputs[0].sum() + position_outputs[3].sum() +
        position_outputs[4].sum()
    )
    position_forces = -torch.autograd.grad(position_total, positions)[0]
    route_error = float((assembled_forces - position_forces).abs().max())
    if route_error > 3.0e-11:
        raise RuntimeError(f"physical force route failed: {route_error}")
    if abs(float(charges.sum())) > 2.0e-12:
        raise RuntimeError("partial charges are not conserved")

    selected_coordinates = [0, 1, 4, 7, 9, 11, 13, 16]
    epsilon = 1.0e-5
    max_fd = 0.0
    fd_checks: list[dict[str, float | int]] = []
    with torch.no_grad():
        for index in selected_coordinates:
            plus = positions0.reshape(-1).clone()
            minus = positions0.reshape(-1).clone()
            plus[index] += epsilon
            minus[index] -= epsilon

            def scalar(flat_positions: torch.Tensor) -> torch.Tensor:
                pos = flat_positions.reshape_as(positions0)
                values = evaluate(
                    edge_vectors(pos, sr_receivers, sr_senders),
                    edge_vectors(pos, lr_receivers, lr_senders),
                )
                return values[0].sum() + values[3].sum() + values[4].sum()

            numerical = float((scalar(plus) - scalar(minus)) / (2.0 * epsilon))
            analytic = float(-position_forces.reshape(-1)[index])
            scaled = abs(numerical - analytic) / (1.0 + abs(analytic))
            max_fd = max(max_fd, scaled)
            fd_checks.append({
                "position_flat_index": index,
                "numerical_energy_gradient": numerical,
                "negative_force": analytic,
                "scaled_error": scaled,
            })
    if max_fd > 3.0e-5:
        raise RuntimeError(f"physical force finite difference failed: {max_fd}")

    net_force = position_forces.sum(dim=0)
    torque_positions = positions0
    if box_length is not None:
        anchor = positions0[0:1]
        torque_positions = anchor + edge_vectors(
            positions0, torch.arange(nodes), torch.zeros(nodes, dtype=torch.long)
        )
    torque = torch.linalg.cross(torque_positions, position_forces).sum(dim=0)

    def edge_virial(
        vectors: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        raw = -torch.einsum("ei,ej->ij", vectors, gradients)
        return 0.5 * (raw + raw.T)

    sr_global_virial = edge_virial(base_sr_vectors, sr_edge_gradient)
    lr_global_virial = edge_virial(base_lr_vectors, lr_edge_gradient)
    global_virial = sr_global_virial + lr_global_virial
    virial_antisymmetry = float((
        -torch.einsum("ei,ej->ij", base_sr_vectors, sr_edge_gradient)
        -torch.einsum("ei,ej->ij", base_lr_vectors, lr_edge_gradient)
    ).sub((
        -torch.einsum("ei,ej->ij", base_sr_vectors, sr_edge_gradient)
        -torch.einsum("ei,ej->ij", base_lr_vectors, lr_edge_gradient)
    ).T).abs().max())
    strain_epsilon = 1.0e-6
    with torch.no_grad():
        plus = evaluate(
            base_sr_vectors * (1.0 + strain_epsilon),
            base_lr_vectors * (1.0 + strain_epsilon),
        )
        minus = evaluate(
            base_sr_vectors * (1.0 - strain_epsilon),
            base_lr_vectors * (1.0 - strain_epsilon),
        )
        plus_energy = plus[0].sum() + plus[3].sum() + plus[4].sum()
        minus_energy = minus[0].sum() + minus[3].sum() + minus[4].sum()
        strain_derivative = float(
            (plus_energy - minus_energy) / (2.0 * strain_epsilon)
        )
    virial_trace = float(torch.trace(global_virial))
    strain_scaled_error = abs(strain_derivative + virial_trace) / (
        1.0 + abs(strain_derivative)
    )
    if virial_antisymmetry > 3.0e-10:
        raise RuntimeError(
            f"global virial is not symmetric: {virial_antisymmetry}"
        )
    if strain_scaled_error > 3.0e-6:
        raise RuntimeError(
            f"isotropic strain/virial check failed: {strain_scaled_error}"
        )
    global_virial_lammps6 = torch.stack((
        global_virial[0, 0], global_virial[1, 1], global_virial[2, 2],
        global_virial[0, 1], global_virial[0, 2], global_virial[1, 2],
    ))

    max_net_force = float(net_force.abs().max())
    max_torque = float(torque.abs().max())
    if max_net_force > 3.0e-10 or max_torque > 3.0e-10:
        raise RuntimeError("physical objective invariance failed")

    electrostatic = model.electrostatic_potential
    dispersion_module = model.dispersion_potential
    parameters = {
        "ke": float(electrostatic.ke),
        "electrostatic_sigma": electrostatic_sigma,
        "cutoff": cutoff_lr,
        "electrostatic_cuton": 0.45 * cutoff_lr,
        "fine_structure": float(FINE_STRUCTURE),
        "bohr": float(BOHR),
        "hartree": float(HARTREE),
        "dispersion_scale": dispersion_scale,
        "dispersion_cuton": cutoff_lr - dispersion_damping,
        "pair_scale": float(dispersion_module.c),
    }
    tensors = {
        "positions": positions0,
        "sr_edge_vectors": base_sr_vectors,
        "lr_pair_vectors": base_lr_vectors,
        "sr_distances": sr_lengths,
        "sr_sh_vectors": sh_vectors,
        "lr_distances": lr_lengths,
        "atomic_energies": atomic_energy,
        "partial_charges": charges,
        "hirshfeld_ratios": hirshfeld,
        "charge_gradient": charge_gradient,
        "hirshfeld_gradient": hirshfeld_gradient,
        "lr_pair_radial_gradient": lr_radial_gradient,
        "implicit_sr_edge_gradients": sr_edge_gradient,
        "implicit_sr_atomic_forces": implicit_forces,
        "direct_lr_pair_force_vectors": lr_edge_gradient,
        "direct_lr_atomic_forces": direct_lr_forces,
        "assembled_atomic_forces": position_forces,
        "sr_global_virial_tensor": sr_global_virial,
        "lr_global_virial_tensor": lr_global_virial,
        "global_virial_tensor": global_virial,
        "global_virial_lammps6": global_virial_lammps6,
        "reference_alphas": ALPHAS.to(torch.float64),
        "reference_c6": C6_COEF.to(torch.float64),
    }
    for label, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"nonfinite tensor: {label}")

    fixture: dict[str, object] = {
        "schema": "so3lr-native-physical-lr-cartesian-force-fixture-v1",
        "objective": "learned_energy_plus_electrostatics_plus_qdo_dispersion",
        "geometry": (
            "physical_water_dimer_periodic_minimum_image_dynamic_box"
            if box_length is not None else
            "physical_water_dimer_nonperiodic_separate_sr_lr_graphs"
        ),
        "periodic_box_length": box_length,
        "vector_convention": "receiver_minus_sender",
        "lr_topology": "unique_unordered_half_pairs",
        "nodes": nodes,
        "sr_edges": sr_edges,
        "lr_pairs": lr_pairs,
        "atomic_numbers": atomic_numbers.tolist(),
        "sr_senders": sr_senders.tolist(),
        "sr_receivers": sr_receivers.tolist(),
        "lr_senders": lr_senders.tolist(),
        "lr_receivers": lr_receivers.tolist(),
        "parameters": parameters,
        "learned_energy_total": float(learned_total),
        "electrostatic_energy_total": float(electrostatic_total),
        "dispersion_energy_total": float(dispersion_total),
        "physical_lr_energy_total": float(physical_lr_total),
        "total_energy": float(total_energy),
        "charge_sum": float(charges.sum()),
        "force_route_max_abs_error": route_error,
        "finite_difference_epsilon": epsilon,
        "finite_difference_checks": fd_checks,
        "finite_difference_max_scaled_error": max_fd,
        "maximum_abs_net_force": max_net_force,
        "maximum_abs_net_torque": max_torque,
        "global_virial_trace": virial_trace,
        "global_virial_antisymmetry": virial_antisymmetry,
        "isotropic_strain_epsilon": strain_epsilon,
        "isotropic_strain_energy_derivative": strain_derivative,
        "isotropic_strain_virial_scaled_error": strain_scaled_error,
    }
    fixture.update({name: flat(value) for name, value in tensors.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print(f"nodes={nodes}")
    print(f"sr_edges={sr_edges}")
    print(f"lr_pairs={lr_pairs}")
    print(f"periodic_box_length={box_length}")
    print(f"learned_energy_total={float(learned_total):.17g}")
    print(f"electrostatic_energy_total={float(electrostatic_total):.17g}")
    print(f"dispersion_energy_total={float(dispersion_total):.17g}")
    print(f"force_route_max_abs_error={route_error:.17g}")
    print(f"finite_difference_max_scaled_error={max_fd:.17g}")
    print(f"global_virial_trace={virial_trace:.17g}")
    print(f"global_virial_antisymmetry={virial_antisymmetry:.17g}")
    print(f"isotropic_strain_virial_scaled_error={strain_scaled_error:.17g}")
    print("pytorch_periodic_global_virial_reference=PASS")
    print("pytorch_physical_long_range_reference=PASS")
    print("pytorch_full_so3lr_cartesian_force_reference=PASS")
    print("translation_rotation_invariance=PASS")


if __name__ == "__main__":
    main()
