#!/usr/bin/env python3
"""Generate independent LAMMPS image and velocity-seed variables."""

from __future__ import print_function

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images", required=True, type=int)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=104729)
    args = parser.parse_args()
    if args.images < 2 or args.images > 256:
        raise RuntimeError("images must be between 2 and 256")
    if args.group < 0:
        raise RuntimeError("group must be non-negative")

    labels = ["{:04d}".format(index) for index in range(args.images)]
    seeds = [args.seed_base + 10007 *
             (args.group * args.images + index)
             for index in range(args.images)]
    with open(args.template, "r") as handle:
        text = handle.read()
    text = text.replace(
        "@IMAGE_VARIABLE@", "variable        IMAGE world " + " ".join(labels))
    text = text.replace(
        "@SEED_VARIABLE@", "variable        SEED world " +
        " ".join(str(seed) for seed in seeds))
    if "@IMAGE_VARIABLE@" in text or "@SEED_VARIABLE@" in text:
        raise RuntimeError("template substitution failed")
    with open(args.output, "w") as handle:
        handle.write(text)
    with open(args.manifest, "w") as handle:
        handle.write("image,seed\n")
        for label, seed in zip(labels, seeds):
            handle.write("{},{}\n".format(label, seed))
    print("images={}".format(args.images))
    print("input={}".format(os.path.realpath(args.output)))
    print("manifest={}".format(os.path.realpath(args.manifest)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("SO3LR_TURBO_EXAMPLE_INPUT=FAIL reason=" + str(error),
              file=sys.stderr)
        sys.exit(1)
