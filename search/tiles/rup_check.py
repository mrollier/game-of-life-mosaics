"""
Self-contained forward DRAT/RUP proof checker (no RAT support needed for
our proofs: CaDiCaL's blocking-clause UNSAT proofs are RUP-checkable, and
any lemma that fails the RUP check is reported rather than accepted).

Deliberately shares no code with the encoder, the enumerator, or the
solution verifier. Intended for the small certification proofs (levels
3-5); use the standard drat-trim tool for level-6-scale proofs.

Usage: python rup_check.py formula.cnf proof.drat
Exit status 0 iff the proof derives the empty clause (or a top-level
conflict) from the formula by reverse unit propagation.
"""

import sys


def parse_dimacs(path):
    clauses = []
    nvars = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("c", "p")):
                if line.startswith("p cnf"):
                    nvars = int(line.split()[2])
                continue
            lits = [int(x) for x in line.split()]
            assert lits[-1] == 0
            clauses.append(tuple(lits[:-1]))
    return nvars, clauses


def parse_proof(path):
    steps = []  # (is_delete, clause tuple)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            delete = line.startswith("d ")
            body = line[2:] if delete else line
            lits = [int(x) for x in body.split()]
            assert lits[-1] == 0
            steps.append((delete, tuple(lits[:-1])))
    return steps


class Checker:
    """Two-watched-literal unit propagation over a growing clause set."""

    def __init__(self, nvars):
        self.val = [0] * (2 * nvars + 2)   # index lit -> 0 unknown, 1 t, -1 f
        self.clauses = {}                  # idx -> list of lits
        self.watches = {}                  # lit -> [idx, ...]
        self.by_key = {}                   # sorted tuple -> [idx, ...]
        self.trail = []
        self.next_idx = 0
        self.top_conflict = False

    def _lit_i(self, lit):
        return 2 * abs(lit) + (lit < 0)

    def value(self, lit):
        return self.val[self._lit_i(lit)]

    def set_true(self, lit):
        self.val[self._lit_i(lit)] = 1
        self.val[self._lit_i(-lit)] = -1
        self.trail.append(lit)

    def unset(self, lit):
        self.val[self._lit_i(lit)] = 0
        self.val[self._lit_i(-lit)] = 0

    def add_clause(self, clause):
        idx = self.next_idx
        self.next_idx += 1
        lits = list(clause)
        self.clauses[idx] = lits
        self.by_key.setdefault(tuple(sorted(lits)), []).append(idx)
        if not lits:
            self.top_conflict = True
            return
        # Prefer unassigned/true literals as watches.
        lits.sort(key=lambda l: self.value(l), reverse=True)
        for w in lits[:2]:
            self.watches.setdefault(w, []).append(idx)
        if len(lits) == 1 or (self.value(lits[0]) != 1 and
                              self.value(lits[1]) == -1):
            # unit (or falsified-but-for-one) under current assignment
            if self.value(lits[0]) == -1:
                self.top_conflict = True
            elif self.value(lits[0]) == 0:
                pos = len(self.trail)
                self.set_true(lits[0])
                if not self._propagate_from(pos):
                    self.top_conflict = True

    def delete_clause(self, clause):
        key = tuple(sorted(clause))
        idxs = self.by_key.get(key)
        if not idxs:
            return  # deleting an unknown clause is harmless for soundness
        idx = idxs.pop()
        self.clauses.pop(idx, None)  # lazily skipped in watch lists

    def _propagate_from(self, start):
        """Unit propagation over the trail suffix; False on conflict."""
        pos = start
        while pos < len(self.trail):
            lit = self.trail[pos]
            pos += 1
            falsified = -lit
            watchlist = self.watches.get(falsified, [])
            new_watchlist = []
            for wi, idx in enumerate(watchlist):
                lits = self.clauses.get(idx)
                if lits is None:
                    continue  # deleted
                # ensure falsified is at position 1
                if len(lits) >= 2 and lits[0] == falsified:
                    lits[0], lits[1] = lits[1], lits[0]
                if len(lits) >= 2 and self.value(lits[0]) == 1:
                    new_watchlist.append(idx)  # satisfied
                    continue
                moved = False
                for k in range(2, len(lits)):
                    if self.value(lits[k]) != -1:
                        lits[1], lits[k] = lits[k], lits[1]
                        self.watches.setdefault(lits[1], []).append(idx)
                        moved = True
                        break
                if moved:
                    continue
                new_watchlist.append(idx)
                other = lits[0]
                v = self.value(other)
                if v == -1:
                    self.watches[falsified] = \
                        new_watchlist + watchlist[wi + 1:]
                    return False
                if v == 0:
                    self.set_true(other)
            self.watches[falsified] = new_watchlist
        return True

    def check_lemma(self, clause):
        """RUP check: assuming every literal false must yield a conflict."""
        mark = len(self.trail)
        conflict = False
        for lit in clause:
            v = self.value(lit)
            if v == 1:
                conflict = True  # clause already satisfied at top level
                break
            if v == 0:
                self.set_true(-lit)
        if not conflict:
            conflict = not self._propagate_from(mark)
        for lit in self.trail[mark:]:
            self.unset(lit)
        del self.trail[mark:]
        return conflict

    def add_persistent(self, clause):
        """Add a verified lemma; propagate persistently if it became unit."""
        self.add_clause(clause)


def main():
    formula_path, proof_path = sys.argv[1], sys.argv[2]
    nvars, clauses = parse_dimacs(formula_path)
    steps = parse_proof(proof_path)
    ch = Checker(nvars)
    for c in clauses:
        ch.add_persistent(c)
        if ch.top_conflict:
            print("VERIFIED (formula has a top-level conflict)")
            return 0
    for i, (delete, clause) in enumerate(steps):
        if delete:
            ch.delete_clause(clause)
            continue
        if not ch.check_lemma(clause):
            print(f"FAILED at proof step {i + 1}: {clause} is not RUP")
            return 1
        ch.add_persistent(clause)
        if ch.top_conflict or not clause:
            print(f"VERIFIED: empty clause derived after {i + 1} steps "
                  f"({sum(1 for d, _ in steps if not d)} lemmas total)")
            return 0
    print("FAILED: proof ends without deriving the empty clause")
    return 1


if __name__ == "__main__":
    sys.exit(main())
