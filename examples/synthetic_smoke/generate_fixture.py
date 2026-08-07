"""Generate the tiny deterministic fixture used by the smoke test."""

from pathlib import Path

import numpy as np


def main() -> None:
    output = Path(__file__).with_name("synthetic_signal.npy")
    np.save(output, np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float64))
    print(output)


if __name__ == "__main__":
    main()
