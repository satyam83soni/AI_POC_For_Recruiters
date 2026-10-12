"""
Remove every learning comment (lines starting with `# >> `) from the project.

    python scripts/strip_notes.py          show what would be removed
    python scripts/strip_notes.py --apply  actually remove them

Only needed if you changed code on the `annotated` branch and want to keep
those changes. Otherwise just `git checkout master` -- it has no notes.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MARKER = "# >> "


def main() -> int:
    apply = "--apply" in sys.argv
    total = 0
    for path in sorted(ROOT.glob("app/*.py")) + sorted(ROOT.glob("scripts/*.py")):
        if path.name == "strip_notes.py":
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if not line.lstrip().startswith(MARKER)]
        removed = len(lines) - len(kept)
        if removed:
            total += removed
            print(f"{removed:>4} notes  {path.relative_to(ROOT)}")
            if apply:
                path.write_text("".join(kept), encoding="utf-8")
    print(f"\n{total} note lines {'removed' if apply else 'found (run with --apply to remove)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
