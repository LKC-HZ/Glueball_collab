import numpy as np
from itertools import permutations

'''
THIS MODULE GENERATES ALL VALID VERTEX CONFIGURATIONS FOR A GENERAL
QUARK / ANTIQUARK / GLUON PROCESS IN COLOR SPACE.

A "particle" is represented as (num, ptype) where:
    num   : 1-based particle ID
    ptype : 'q' (quark), 'qb' (antiquark), or 'g' (gluon)

The partition algorithm itself is particle-agnostic; only the vertex classifier
looks at ptype.
'''

# ------------------------------------------------------------------
# Core partition (unchanged algorithm, generic over particle type)
# ------------------------------------------------------------------

def partition(particles):
    """Divide a list of particles into all possible subset splittings.
    Each subset has size <= 3 (max legs on a QCD vertex).
    """
    n = len(particles)
    result = []

    def go(index, current):
        if index == n:
            result.append([subset[:] for subset in current])
            return

        elem = particles[index]

        # choice 1: start a new subset
        current.append([elem])
        go(index + 1, current)
        current.pop()

        # choice 2: add to an existing subset (max size 2 before adding -> 3)
        for i in range(len(current)):
            if len(current[i]) <= 2:
                current[i].append(elem)
                go(index + 1, current)
                current[i].pop()

    go(0, [])
    return result


# ------------------------------------------------------------------
# Vertex classification
# ------------------------------------------------------------------

def _ptype(p):
    """Extract particle type string from a particle tuple."""
    return p[1]


def get_vertex_type(inc, out):
    """
    Classify a vertex by the particle content of its incoming / outgoing legs.

    Parameters
    ----------
    inc, out : list of (num, ptype)
        Incoming and outgoing particles for this vertex.

    Returns
    -------
    str : one of
        'spectator', '3-gluon', '4-gluon',
        'quark-gluon', 'antiquark-gluon', 'gluon-splitting', 'invalid'
    """
    all_p = [_ptype(e) for e in inc + out]

    n_q  = sum(1 for t in all_p if t == 'q')
    n_qb = sum(1 for t in all_p if t == 'qb')
    n_g  = sum(1 for t in all_p if t == 'g')

    # ---- spectator : 1 -> 1, same species --------------------------------
    if len(inc) == 1 and len(out) == 1 and _ptype(inc[0]) == _ptype(out[0]):
        return "spectator"

    # ---- 3-gluon ---------------------------------------------------------
    if n_q == 0 and n_qb == 0 and n_g == 3:
        return "3-gluon"

    # ---- 4-gluon ---------------------------------------------------------
    if n_q == 0 and n_qb == 0 and n_g == 4:
        return "4-gluon"

    # ---- quark-gluon  (q -> q g  or  q g -> q) ---------------------------
    # Total: 2 quarks + 1 gluon
    if n_q == 2 and n_qb == 0 and n_g == 1:
        return "quark-gluon"

    # ---- antiquark-gluon  (qb -> qb g  or  qb g -> qb) -------------------
    # Total: 2 antiquarks + 1 gluon
    if n_q == 0 and n_qb == 2 and n_g == 1:
        return "antiquark-gluon"

    # ---- gluon splitting  (g -> q qb  or  q qb -> g) ---------------------
    # Total: 1 quark + 1 antiquark + 1 gluon
    if n_q == 1 and n_qb == 1 and n_g == 1:
        return "gluon-splitting"

    return "invalid"


# ------------------------------------------------------------------
# Pair partitions and generate plans
# ------------------------------------------------------------------

def pair_partitions(inc_partition, out_partition):
    """Pair incoming subsets with outgoing subsets via permutations.
    Only keep pairings where every matched subset forms a valid vertex.
    """
    if len(inc_partition) != len(out_partition):
        return []

    valid_pairings = []
    n = len(inc_partition)

    for perm in permutations(range(n)):
        pairing = []
        ok = True

        for i in range(n):
            inc_subset = inc_partition[i]
            out_subset = out_partition[perm[i]]
            vtype = get_vertex_type(inc_subset, out_subset)

            if vtype == "invalid":
                ok = False
                break

            pairing.append({
                'incoming': inc_subset,
                'outgoing': out_subset,
                'type': vtype,
            })

        if ok:
            valid_pairings.append(pairing)

    return valid_pairings


def generate_vertex_plans(inc_particles, out_particles):
    """
    Generate every valid vertex plan for a process.

    Parameters
    ----------
    inc_particles : list of (num, ptype)
        Incoming particle list.
    out_particles : list of (num, ptype)
        Outgoing particle list.

    Returns
    -------
    List of dicts:
        {'id': int,
         'vertices': [{'incoming': [...], 'outgoing': [...], 'type': str}, ...]}
    """
    inc_partitions = partition(inc_particles)
    out_partitions = partition(out_particles)

    all_plans = []
    plan_id = 1

    for inc_p in inc_partitions:
        for out_p in out_partitions:
            pairings = pair_partitions(inc_p, out_p)
            for pairing in pairings:
                all_plans.append({'id': plan_id, 'vertices': pairing})
                plan_id += 1

    return all_plans


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _pid_str(p):
    """String representation of a particle's numeric ID."""
    return str(p[0])


def format_plan(plan):
    """Pretty-print a single plan (for debugging)."""
    lines = [f"config {plan['id']}:\n"]
    for v_idx, v in enumerate(plan['vertices'], start=1):
        inc_str = ','.join(_pid_str(x) for x in v['incoming'])
        out_str = ','.join(_pid_str(x) for x in v['outgoing'])
        lines.append(
            f"    vertex {v_idx}: [{inc_str}] -> [{out_str}] ; {v['type']}\n"
        )
    return "".join(lines)


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # ---- TEST 1 : q g -> q g g  (quark-gluon scattering) ----------------
    print("=" * 60)
    print("TEST 1: q g -> q g g")
    print("=" * 60)
    inc = [(1, 'q'), (2, 'g')]
    out = [(1, 'q'), (2, 'g'), (3, 'g')]
    plans = generate_vertex_plans(inc, out)
    print(f"  Total configs: {len(plans)}")

    # Count vertex-type frequencies across all plans
    vtype_counts = {}
    for p in plans:
        for v in p['vertices']:
            vt = v['type']
            vtype_counts[vt] = vtype_counts.get(vt, 0) + 1
    print("  Vertex-type histogram:")
    for vt, c in sorted(vtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {vt:20s}: {c}")

    print("  First 6 plans:")
    for p in plans[:6]:
        print(format_plan(p))
    print()

    # ---- TEST 2 : g -> q qb  (gluon splitting) --------------------------
    print("=" * 60)
    print("TEST 2: g -> q qb")
    print("=" * 60)
    plans_split = generate_vertex_plans([(1, 'g')], [(1, 'q'), (2, 'qb')])
    print(f"  Total configs: {len(plans_split)}")
    for p in plans_split:
        print(format_plan(p))
    print()

    # ---- TEST 3 : g g -> g g g  (pure gluon, reference value = 6) ------
    print("=" * 60)
    print("TEST 3: g g -> g g g  (reference: 6 configs)")
    print("=" * 60)
    start = time.perf_counter()
    plans_g = generate_vertex_plans(
        [(1, 'g'), (2, 'g')],
        [(1, 'g'), (2, 'g'), (3, 'g')]
    )
    end = time.perf_counter()
    print(f"  Total configs: {len(plans_g)}  (expected 6)")
    print(f"  Wall time: {end - start:.4f} s")
    for p in plans_g:
        print(format_plan(p))
    print()

    # ---- TEST 4 : Performance sweep (pure gluon) ------------------------
    print("=" * 60)
    print("TEST 4: Performance sweep (pure gluon)")
    print("=" * 60)
    for n in range(2, 6):
        for m in range(2, 6):
            inc_g = [(i, 'g') for i in range(1, n + 1)]
            out_g = [(i, 'g') for i in range(1, m + 1)]
            t0 = time.perf_counter()
            test = generate_vertex_plans(inc_g, out_g)
            t1 = time.perf_counter()
            print(f"  {n}g -> {m}g:  {len(test):6d} configs  ({t1 - t0:.4f} s)")
    print("=" * 60)
