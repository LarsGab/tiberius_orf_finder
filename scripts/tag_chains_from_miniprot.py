"""Tag every intron / start_codon / stop_codon in miniprot_scored.gff with
`chain_id=<Parent>_<prot>` — a chainedHints.py equivalent that skips the
hc.gff high-confidence filter.

Output has the same schema as chained_hints.gff so it plugs into
prepare_hint_rescue_loci.py --chained_hints unchanged.

Usage:
    python tag_chains_from_miniprot.py miniprot_scored.gff out.gff
"""

import re
import sys


HINT_TYPES = {"intron", "start_codon", "stop_codon"}


def _attr(col: str, key: str) -> str | None:
    m = re.search(rf"{key}=([^;\s]+)", col)
    return m.group(1) if m else None


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    in_path, out_path = sys.argv[1], sys.argv[2]

    kept = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2].lower() not in HINT_TYPES:
                continue
            parent = _attr(c[8], "Parent")
            prot   = _attr(c[8], "prot")
            if not parent or not prot:
                continue
            chain_id = f"{parent}_{prot}"
            # Append chain_id to attributes if not already present
            c[8] = c[8].rstrip() + (";" if not c[8].rstrip().endswith(";") else "") \
                 + f"chain_id={chain_id};"
            fout.write("\t".join(c) + "\n")
            kept += 1
    print(f"wrote {kept} tagged hint lines", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
