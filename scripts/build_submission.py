"""Build a Kaggle-ready single-file main.py from modular source files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "orbit_agent" / "core.py"
OUT = ROOT / "main.py"


def main() -> None:
    OUT.write_text(CORE.read_text())
    print(f"Wrote {OUT.relative_to(ROOT)} from {CORE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
