from pathlib import Path
from lammps_mace_calculator import pipe, run_lammps_mace


def main():
    root = Path(__file__).resolve().parent

    run_lammps_mace(
        str(root / "CO_art_str.lmp"),
        str(root / "model-mliap.pt"),
    )

    pipe.submit()


if __name__ == "__main__":
    main()
