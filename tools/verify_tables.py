#!/usr/bin/env python3
"""Check the open tables (distortkit/tables/*.json) against a copy of the ISOTROPY data tables.

    python tools/verify_tables.py /path/to/iso_directory     (the folder holding data_space.txt and data_wyckoff.txt)

For every one of the 230 space groups: the operations are compared as sets; the Wyckoff positions are compared one
by one (same letter <=> same special position, i.e. the same affine subspace up to the space group and lattice
translations, and the same multiplicity).  Needs only numpy; the ISOTROPY files are read, never copied.
"""
import itertools
import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TWO_ORIGIN = {48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134, 137, 138, 141, 142, 201, 203, 222, 224, 227, 228}
sys.path.insert(0, HERE)
import isotropy_reader as iso                        # noqa: E402
sys.path.insert(0, os.path.join(HERE, ".."))
from build_tables import Subspace, nullspace, rref   # noqa: E402


def parse(s):
    return Fraction(s)


def subspace_from_template(const, rows):
    """Subspace of a (const + x*row0 + y*row1 + z*row2) template (rows with free parameter directions)."""
    dirs = [tuple(parse(v) for v in r) for r in rows if any(parse(v) != 0 for v in r)]
    return Subspace([parse(v) for v in const], rref(dirs)[0] if dirs else [])


def orbit_keys(S, ops):
    return {S.image(R, t).key() for R, t in ops}


def permute(S, P):
    """Subspace with the coordinates permuted: new coordinate i = old coordinate P[i]."""
    p = [S.p[P[i]] for i in range(3)]
    dirs = [tuple(d[P[i]] for i in range(3)) for d in S.dirs]
    return Subspace(p, rref(dirs)[0] if dirs else [])


F = Fraction
PRIM = {   # columns = primitive basis vectors in conventional coordinates
    "P": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "A": [[1, 0, 0], [0, F(1, 2), F(-1, 2)], [0, F(1, 2), F(1, 2)]],
    "B": [[F(1, 2), 0, F(-1, 2)], [0, 1, 0], [F(1, 2), 0, F(1, 2)]],
    "C": [[F(1, 2), F(-1, 2), 0], [F(1, 2), F(1, 2), 0], [0, 0, 1]],
    "I": [[F(-1, 2), F(1, 2), F(1, 2)], [F(1, 2), F(-1, 2), F(1, 2)], [F(1, 2), F(1, 2), F(-1, 2)]],
    "F": [[0, F(1, 2), F(1, 2)], [F(1, 2), 0, F(1, 2)], [F(1, 2), F(1, 2), 0]],
    "R": [[F(2, 3), F(-1, 3), F(-1, 3)], [F(1, 3), F(1, 3), F(-2, 3)], [F(1, 3), F(1, 3), F(1, 3)]],
}


def mat_inv(M):
    return np.linalg.inv(np.array(M, dtype=float))


def apply_basis(S, A):
    """Subspace S given in primitive coordinates -> conventional coordinates with x_conv = A x_prim (A is 3x3 Fractions)."""
    def mv(v):
        return tuple(sum(Fraction(A[i][j]) * v[j] for j in range(3)) for i in range(3))
    dirs = [mv(d) for d in S.dirs]
    return Subspace(mv(S.p), rref(dirs)[0] if dirs else [])


def basis_candidates(letter, widen=False):
    """Conversions primitive -> conventional coordinates to try.  ``widen``: every centring type (permuting the axes
    turns C into A or B)."""
    out = {}
    for L in (PRIM if widen else [letter]):
        A0 = [[Fraction(v) for v in row] for row in PRIM[L]]
        cols = [[A0[i][j] for i in range(3)] for j in range(3)]           # as listed: rows = primitive vectors
        cand = {f"{L}: as listed (rows are vectors)": A0, f"{L}: transposed (columns are vectors)": cols}
        for name, M in list(cand.items()):
            inv = mat_inv(M)
            cand[name + ", inverse"] = [[Fraction(float(v)).limit_denominator(12) for v in row] for row in inv]
        out.update(cand)
    return out


def shifted(S, s):
    return Subspace([S.p[i] + s[i] for i in range(3)], S.dirs)


def main(isodir, tables):
    iso.set_data_dir(isodir)
    G = iso.space_groups()
    sg = json.load(open(os.path.join(tables, "space_groups.json")))["groups"]
    wy = json.load(open(os.path.join(tables, "wyckoff.json")))["groups"]
    W = iso.get_tables("wyckoff")
    cnt, ptr = iso.ints(W, "iwyckoff_count"), iso.ints(W, "iwyckoff_pointer")
    fr = iso.ints(W, "iwyckoff_fract").reshape(-1, 4, 4)
    labels = [s.strip() for s in iso.strings(W, "wyckoff_label")]
    bad_ops, bad_wy, permuted_groups, letter_diff = [], [], {}, {}
    total = 0
    for n in range(1, 231):
        a = {(tuple(int(x) for x in R.ravel()), tuple(t)) for R, t in G[n].ops}
        rec = sg[str(n)]
        b = {(tuple(R), tuple(Fraction(v) for v in t)) for R, t in rec["ops"]}
        if a != b:
            bad_ops.append(n)
            continue
        ops = [(tuple(R), tuple(Fraction(v) for v in t)) for R, t in rec["ops"]]
        theirs = {}
        base = int(ptr[n - 1]) - 1
        for j in range(int(cnt[n - 1])):
            f = fr[base + j]
            const = [Fraction(int(f[0][k]), int(f[0][3])) for k in range(3)]
            rows = [[Fraction(int(f[r][k]), int(f[r][3])) if f[r][3] else Fraction(0) for k in range(3)] for r in (1, 2, 3)]
            theirs[labels[base + j].replace("alpha", "A")] = subspace_from_template([str(c) for c in const], [[str(v) for v in r] for r in rows])
        ours = {}
        for item in wy[str(n)]:
            ours[item["letter"]] = (orbit_keys(subspace_from_template(item["const"], item["rows"]), ops), item["multiplicity"])
        total += len(theirs)
        problems = []
        if set(theirs) != set(ours):
            problems.append(f"letters differ: ISOTROPY {sorted(theirs)} vs ours {sorted(ours)}")
        else:
            def mismatches(perm):
                return [L for L in theirs if orbit_keys(permute(theirs[L], perm), ops) != ours[L][0]]
            bad = mismatches((0, 1, 2))
            if bad:                                               # ISOTROPY's table may use another setting: axes, primitive basis, origin
                letter = rec["symbol"][0]
                shifts = [(Fraction(0),) * 3]
                if n in TWO_ORIGIN:
                    shifts = [(Fraction(a, 8), Fraction(b, 8), Fraction(c, 8)) for a in range(8) for b in range(8) for c in range(8)]
                done = False
                for widen in (False, True):
                    if widen and n in TWO_ORIGIN:
                        break
                    if widen and n in (151, 153, 154):                # 3_1/3_2 screw axes: the origin can differ along z
                        shifts = [(Fraction(0), Fraction(0), Fraction(c, 24)) for c in range(24)]
                    for cname, A in basis_candidates(letter, widen).items():
                        base_sub = {L: apply_basis(S, A) for L, S in theirs.items()}
                        for perm in itertools.permutations(range(3)):
                            for sh in shifts:
                                ok = all(orbit_keys(shifted(permute(base_sub[L], perm), sh), ops) == ours[L][0] for L in theirs)
                                if ok:
                                    permuted_groups[n] = (cname, perm, tuple(str(v) for v in sh))
                                    done = True
                                    break
                            if done:
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    bad = []
            if bad:
                ours_classes = {frozenset(v[0]) for v in ours.values()}
                theirs_classes = {frozenset(orbit_keys(S, ops)) for S in theirs.values()}
                if ours_classes == theirs_classes:
                    swaps = {L: next(L2 for L2, v in ours.items() if frozenset(v[0]) == frozenset(orbit_keys(theirs[L], ops))) for L in bad}
                    letter_diff[n] = swaps
                    problems.append("same positions, different letters (ISOTROPY letter -> ours): " + ", ".join(f"{a}->{b}" for a, b in sorted(swaps.items())))
                else:
                    problems.append("different positions: " + ",".join(sorted(bad)))
        if problems:
            bad_wy.append((n, problems))
    print(f"space-group operations identical: {230 - len(bad_ops)} of 230" + (f"   differ: {bad_ops}" if bad_ops else ""))
    print(f"Wyckoff tables identical (letters and positions): {230 - len(bad_wy)} of 230 groups; {total} ISOTROPY positions checked")
    if permuted_groups:
        kinds = {}
        for n, (cname, perm, sh) in permuted_groups.items():
            kinds.setdefault((cname, perm, sh) if n in TWO_ORIGIN or n < 16 else (cname, perm), []).append(n)
        print(f"  of these, {len(permuted_groups)} groups match only after converting ISOTROPY's Wyckoff table to the conventional "
              "cell / our axes / origin:")
        for k, v in sorted(kinds.items(), key=lambda kv: kv[1]):
            print(f"     {len(v):3d} groups  {k[0]}; axes {k[1]}" + (f"; shift {k[2]}" if len(k) > 2 and any(x != '0' for x in k[2]) else ""))
    for n, pr in bad_wy:
        print(f"  group {n}: {'; '.join(pr)[:400]}")
    return not bad_ops and not bad_wy


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ok = main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "distortkit", "tables"))
    sys.exit(0 if ok else 1)
