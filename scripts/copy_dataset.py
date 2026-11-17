"""Create an independent demonstration copy before RECAP writes return metadata."""

import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "meta/info.json").is_file():
        raise ValueError("Not a LeRobot dataset")
    if output == source or output.is_relative_to(source):
        raise ValueError("Choose a separate output")
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output, symlinks=False)
    print(output)


if __name__ == "__main__":
    main()
