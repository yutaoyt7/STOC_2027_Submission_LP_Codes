#region Setup
#region Imports

import gurobipy as gp
from gurobipy import GRB, quicksum 
import time
import gc
import os
import psutil
import sys
import math
import argparse

#endregion

#region Parameters
# parser = argparse.ArgumentParser()
# parser.add_argument("--n", type=int, default=5)
# parser.add_argument("--odd_girth", type=int, default=3)
# args = parser.parse_args()

n = 5
odd_girth = 3
n_pairing = n // 2

# if n < 1:
#     parser.error("--n must be a positive integer")
# if odd_girth < 3 or odd_girth % 2 != 1:
#     parser.error("--odd_girth must be an odd integer >= 3")
#endregion

#region Gurobi model and solver parameters
model = gp.Model("GurobiLP")
model.setParam("Method", 2)          # Barrier method (best for large continuous LPs)
model.setParam("Crossover", 0)       # Skip crossover to save time (if you only need the LP optimum)
model.setParam("Threads", 32)        # Match to physical cores on one socket
model.setParam("Presolve", 2)        # Aggressive presolve
model.setParam("BarOrder", 1)        # Use approximate minimum degree ordering (usually best)
model.setParam("BarConvTol", 1e-6)   # Tight barrier convergence tolerance
model.setParam("OptimalityTol", 1e-5) # Default or slightly tighter if needed
model.setParam("OutputFlag", 1)      # Keep log output visible

#endregion

#endregion

#region 5. LP Setting and Coarse Profile Probabilities
#region 5.1 Coarse Profile Probability
# Definition 5.3 (V1): final matched rank and optimum-partner rank.
P_u_vp_us= {
    (u, vp, us) : model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for vp in range(1, n + 1)
    for us in range(1, n + 1)
}

# Definition 5.3 (V1): final matched rank and backup rank.
P_u_vp_bp= {
    (u, vp, bp) : model.addVar(ub = 1 / n) 
    for u in range(1, n + 1)
    for vp in range(1, n + 1)
    for bp in range(vp, n + 2)
}

# Definition 5.3 (V1): coarse profiles with u* removed.
coarse_profile_unmatched = {
    u: model.addVar(ub = 1 / n) 
    for u in range(1, n + 1)
}
coarse_profile_no_backup = {
    (u, v):  model.addVar(ub = 1 / n) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
}
coarse_profile_backup = {
    (u, v, b):  model.addVar(ub = 1 / n) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for b in range(v, n + 1)
}

# Constraint 5.4 (Objective): minimize total matched probability.
model.setObjective(gp.quicksum(P_u_vp_bp.values()), GRB.MINIMIZE)

# Constraint 5.5 (C1): coarse-profile partition; bounds set at creation.
for u in range(1, n + 1):
    total_prob_u = gp.LinExpr()
    total_prob_u += coarse_profile_unmatched[u]
    for v in range(1, n + 1):
        total_prob_u += coarse_profile_no_backup[u, v]
        for b in range(v, n + 1):
            total_prob_u += coarse_profile_backup[u, v, b]
    model.addConstr(total_prob_u == 1/n)

# Constraint 5.5 (C2): consistency of the two final-match marginals.
for u in range(1, n + 1):
    for vp in range(1, n + 1):
        model.addConstr(
            gp.quicksum(
                P_u_vp_bp[u, vp, bp]
                for bp in range(vp, n + 2)
            )
            == gp.quicksum(
                P_u_vp_us[u, vp, us]
                for us in range(1, n + 1)
            )
        )

#endregion

#endregion

#region 6. Detailed Profile Probabilities
#region 6.1 Detailed Profiles Partition of the Unmatched Coarse Profile
# Definition 6.1 (V2): detailed unmatched profiles.
profile_unmatched_us_vs_u_bot = {
    (u, us, vs):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for us in range(1, n + 1)
    for vs in range(1, u + 1)
}
profile_unmatched_us_vs_u_ANY = {
    (u, us, vs):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for us in range(1, n + 1)
    for vs in range(1, u + 1)
}
profile_unmatched_us_pf = {
    (u, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for us in range(1, n + 1)
}

# Constraint 6.2 (C3): unmatched coarse-profile partition.
for u in range(1, n + 1):
    for us in range(1, n + 1):
        total_prob = gp.LinExpr()
        for vs in range (1 , u + 1):
            total_prob += profile_unmatched_us_vs_u_bot[u, us, vs]
            total_prob += profile_unmatched_us_vs_u_ANY[u, us, vs]
        total_prob += profile_unmatched_us_pf[u, us]
        model.addConstr(total_prob == 1 / n * coarse_profile_unmatched[u]) 

#endregion

#region 6.2 Detailed Profiles Partition of the No-Backup Coarse Profile
# Definition 6.3 (V3), cases 1-6: no-backup detailed profiles.
profile_no_backup_us_pf = {
    (u, v, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for us in range(1, v + 1)
}
profile_no_backup_us_bot = {
    (u, v, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for us in range(v, n + 1)
}
profile_no_backup_u_NO_WORSE = {
    (u, v, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for us in range(1, n + 1)
}

profile_no_backup_SHORT_u_v_us = {
    (u, v, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, u + 1)
    for us in range(1, u + 1)
}
profile_no_backup_EQUAL_u_v_us_vs_u2 = {
    (u, v, us, vs, u2):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for us in range(1, u + 1)
    for vs in range(1, min(u,v) + 1)
    for u2 in range(us, u + 1)
}
profile_no_backup_LARGER_u_v_us_vs_u2 = {
    (u, v, us, vs, u2):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for us in range(1, u + 1)
    for vs in range(1, min(u,v) + 1)
    for u2 in range(us, u)
}

# Definition 6.3 (V3), case 7: OBLIVIOUS profiles.
# Constraint 6.5: untagged zero-support rule is encoded by ub=0.
profile_no_backup_us_vs_OBLIVIOUS = {
    (u, v, us, vs):  model.addVar(ub=0 if us < v and vs > u else 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for us in range(1, n + 1)
    for vs in range(1, n + 1)
} 

# Definition 6.3, case 8: aggregate WORSE profiles.
profile_no_backup_WORSE_u_ANY_us_vs = {
    (u, us, vs):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for us in range(1, u + 1)
    for vs in range(1, u + 1)
}

# Constraint 6.4 (C4): no-backup coarse-profile partition.
for u in range(1, n + 1):
    for v in range(1, n + 1):
        for us in range(1, n + 1):
            total_prob=gp.LinExpr()
            if us <=v:
                total_prob+=profile_no_backup_us_pf[u, v, us]
            if us >= v:
                total_prob+=profile_no_backup_us_bot[u, v, us]
            total_prob+=profile_no_backup_u_NO_WORSE[u, v, us]

            if v <= u and us <=u:
                total_prob += profile_no_backup_SHORT_u_v_us[u, v, us]
            if us <= u:
                total_prob += gp.quicksum(
                    profile_no_backup_EQUAL_u_v_us_vs_u2[ u, v, us, vs, u2] 
                    for vs in range(1, min(u,v) + 1) 
                    for u2 in range(us, u + 1)
                )
                total_prob += gp.quicksum(
                    profile_no_backup_LARGER_u_v_us_vs_u2[ u, v, us, vs, u2] 
                    for vs in range(1, min(u,v) + 1) 
                    for u2 in range(us, u)
                )
            model.addConstr(total_prob == 1 / n * coarse_profile_no_backup[u, v])

# Constraint 6.5 (C5): u* is either matched or unmatched.
for u in range(1, n + 1):
    for v in range(1, n + 1):
        for us in range(1, n + 1):
            if us < v:
                total_prob = gp.quicksum(
                    profile_no_backup_us_vs_OBLIVIOUS[u, v, us, vs]
                    for vs in range(1, u + 1)
                )
                model.addConstr(total_prob == 1 / n * coarse_profile_no_backup[u, v])
            else:
                total_prob = gp.quicksum(
                    profile_no_backup_us_vs_OBLIVIOUS[u, v, us, vs]
                    for vs in range(1, n + 1)
                )
                total_prob += profile_no_backup_us_bot[u, v, us]
                model.addConstr(total_prob == 1 / n * coarse_profile_no_backup[u, v])

# Constraint 6.5 (C6): elementary OBLIVIOUS lower bounds.
for u in range(1, n + 1):
    for v in range(1, n + 1):
        for us in range(1, n + 1):
            for vs in range(1, n + 1):
                if us < v and vs > u:
                    continue
                total_prob = gp.LinExpr()
                if  us <= v and vs == u:
                    total_prob += profile_no_backup_us_pf[u, v, us]
                if us<=u and vs == v:
                    total_prob += profile_no_backup_SHORT_u_v_us[u, v, us]
                if vs<= v and us <= u:
                    total_prob += gp.quicksum(
                        profile_no_backup_EQUAL_u_v_us_vs_u2[u, v, us, vs, u2]
                        for u2 in range(us, u+ 1)
                    )
                    total_prob += gp.quicksum(
                        profile_no_backup_LARGER_u_v_us_vs_u2[u, v, us, vs, u2]
                        for u2 in range(us, u)
                    )
                model.addConstr(profile_no_backup_us_vs_OBLIVIOUS[u, v, us, vs] >= total_prob)

# Constraint 6.6 (C7): aggregate SHORT, EQUAL, and LARGER events.
for u in range(1, n + 1):
    for us in range(1, u + 1):
        for vs in range(1, u + 1):
            total_prob=gp.LinExpr()
            total_prob+=profile_no_backup_SHORT_u_v_us[u, vs, us]
            for v in range(vs, n + 1): 
                total_prob+= gp.quicksum(
                    profile_no_backup_EQUAL_u_v_us_vs_u2[u, v, us, vs, u2]
                    for u2 in range(us, u+1)
                )  
                total_prob+= gp.quicksum(
                    profile_no_backup_LARGER_u_v_us_vs_u2[u, v, us, vs, u2]
                    for u2 in range(us, u)
                ) 
            model.addConstr(profile_no_backup_WORSE_u_ANY_us_vs[u, us, vs]== total_prob)

#endregion

#region 6.3 Detailed Profiles Partition of the Backup Coarse Profile
# Definition 6.7 (V4): detailed profiles with a backup.
profile_backup_us_pf = {
    (u, v, b, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for b in range(v, n + 1)
    for us in range(1, v + 1)
}
profile_backup_LO_b_us_bot ={
    (u, v, b, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for b in range(v, n + 1)
    for us in range(b, n + 1)
}
profile_backup_HI_b_us_bot ={
    (u, v, b, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for b in range(v, n + 1)
    for us in range(v, b + 1)
}
profile_backup_us_ANY_u_b ={
    (u, v, b, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for b in range(v, n + 1)
    for us in range(1, u + 1)
}
profile_backup_us_ANY_u_ANY ={
    (u, v, b, us):  model.addVar(ub = 1 / (n * n)) 
    for u in range(1, n + 1)
    for v in range(1, n + 1)
    for b in range(v, n + 1)
    for us in range(1, n + 1)
}

# Constraint 6.8 (C8): backup coarse-profile partition.
for u in range(1, n + 1):
    for v in range(1, n + 1):
        for b in range(v, n + 1):
            for us in range(1, n + 1): 
                total_prob=gp.LinExpr()
                if us <= v:
                    total_prob += profile_backup_us_pf[u, v, b, us]
                if us >= b:
                    total_prob += profile_backup_LO_b_us_bot[u, v, b, us] 
                if us >= v and us <= b:
                    total_prob += profile_backup_HI_b_us_bot[u, v, b, us] 
                if us <= u:
                    total_prob += profile_backup_us_ANY_u_b[u, v, b, us]
                total_prob += profile_backup_us_ANY_u_ANY[u, v, b, us]

                model.addConstr( total_prob == 1 / n * coarse_profile_backup[u, v, b])

#endregion

#region 6.4 Finer Profiles for Bad Events
#region Refined SHORT profiles
# Definition 6.9 (V5): refined SHORT profiles.
profile_no_backup_SHORT_u_v_us_bs = {}

for (u, v, us), prob in profile_no_backup_SHORT_u_v_us.items():

    for bs in range(v, u + 1):
        refined_prob = model.addVar(ub=1 / (n * n))

        profile_no_backup_SHORT_u_v_us_bs[u, v, us, bs] = refined_prob


    # Constraint 6.10 (C9): partition each parent SHORT event.
    model.addConstr(
        gp.quicksum(
            profile_no_backup_SHORT_u_v_us_bs[u, v, us, bs]
            for bs in range(v, u + 1)
        ) == prob
    )

#endregion

#region Pairing probabilities
# Definition 6.11: pairing index set.
pairing_indices = {
    (i, j) 
    for i in range(1, n_pairing + 1)
    for j in range(i, n + 1)
}

# Definition 6.12 (V6): EQUAL and LARGER pairing probabilities.
P_PAIRING_EQUAL_u_v_us_vs_u2_usnew = {
    (u, v, us, vs, u2, usnew) : model.addVar( ub = 1 / (n * n))
    for us, usnew in pairing_indices
    for u in range(us, n + 1)
    for v in range(1, usnew + 1)
    for vs in range(1, min(u, v) + 1) 
    for u2 in range(us, min(u, usnew) + 1)
}

P_PAIRING_LARGER_u_v_us_vs_u2_usnew = {
    (u, v, us, vs, u2, usnew) : model.addVar( ub = 1 / (n * n))
    for us, usnew in pairing_indices
    for u in range(us + 1, n + 1)
    for v in range(1, usnew + 1)
    for vs in range(1, min(u, v) + 1) 
    for u2 in range(us, min(u -1, usnew) + 1)
}

# Constraint 6.13 (C10): pairing upper bounds.
for u, v, us, vs, u2, usnew in P_PAIRING_EQUAL_u_v_us_vs_u2_usnew.keys():
    model.addConstr(
        P_PAIRING_EQUAL_u_v_us_vs_u2_usnew[u, v, us, vs, u2, usnew] <= profile_no_backup_EQUAL_u_v_us_vs_u2[u, v, us, vs, u2]
    )

for u, v, us, vs, u2, usnew in P_PAIRING_LARGER_u_v_us_vs_u2_usnew.keys():
    model.addConstr(
        P_PAIRING_LARGER_u_v_us_vs_u2_usnew[u, v, us, vs, u2, usnew] <= profile_no_backup_LARGER_u_v_us_vs_u2[u, v, us, vs, u2]
    )

# Constraint 6.14 (C11): inclusion-exclusion pairing lower bounds.
for us, usnew in pairing_indices:
    for u in range(us, n + 1):
        for v in range(1, usnew + 1):
            LHS = gp.LinExpr()
            RHS = gp.LinExpr()

            LHS += gp.quicksum(
                P_PAIRING_EQUAL_u_v_us_vs_u2_usnew[u, v, us, vs, u2, usnew]
                for vs in range(1, min(u,v) + 1)
                for u2 in range(us, min(u, usnew) + 1)
            )
            LHS += gp.quicksum(
                P_PAIRING_LARGER_u_v_us_vs_u2_usnew[u, v, us, vs, u2, usnew]
                for vs in range(1, min(u,v) + 1)
                for u2 in range(us, min(u - 1, usnew) + 1)
            )

            RHS += profile_no_backup_us_bot[u, v, usnew]
            RHS += gp.quicksum(
                profile_no_backup_EQUAL_u_v_us_vs_u2[u, v, us, vs, u2]
                for vs in range(1, min(u,v) + 1)
                for u2 in range(us, u + 1)
            )
            RHS += gp.quicksum(
                profile_no_backup_LARGER_u_v_us_vs_u2[u, v, us, vs, u2]
                for vs in range(1, min(u,v) + 1)
                for u2 in range(us, u)
            )
            RHS -= 1 / n * coarse_profile_no_backup[u, v]

            model.addConstr(LHS >= RHS)

#endregion

#endregion

#endregion

#region 7. Flow Network
#region 7.1 Canonical Cover of Final Match Profiles
# Definition 7.1: binary interval trees.
def build_binary_interval_tree(L, R):
    children = {}
    def build(left, right):
        block = (left, right)

        if left == right:
            children[block] = ()
            return

        mid = (left + right) // 2

        children[block] = (
            (left, mid),
            (mid + 1, right),
        )

        build(left, mid)
        build(mid + 1, right)

    build(L, R)
    return children

regular_index_tree = build_binary_interval_tree(1, n )
backup_index_tree = build_binary_interval_tree(1, n + 1)

# Definition 7.2: canonical interval cover.
def canonical_cover(tree, root, requested):
    a, b = requested
    L, R = root

    if a == b:
        return [(a,b)]

    cover = []

    def visit(node):
        left, right = node

        if right < a or b < left:
            return

        if a <= left and right <= b:
            cover.append(node)
            return

        for child in tree[node]:
            visit(child)

    visit(root)
    return cover

# Definitions 7.4 and 7.6: eligible canonical box cover.
from itertools import product

def eligible_canonical_box_cover(
    vertex_interval, match_interval, backup_interval
):
    vertex_cover = canonical_cover(
        regular_index_tree, (1, n), vertex_interval
    )
    match_cover = canonical_cover(
        regular_index_tree, (1, n), match_interval
    )
    backup_cover = canonical_cover(
        backup_index_tree, (1, n + 1), backup_interval
    )

    return [
        (vertex_node, match_node, backup_node)
        for vertex_node, match_node, backup_node
        in product(vertex_cover, match_cover, backup_cover)
        if match_node[0] <= backup_node[1]
    ]

# Definitions 7.3 and 7.6: eligible children, splitting the longest coordinate.
def allowed_box_children(box):
    axis = max(
        range(3),
        key=lambda i: box[i][1] - box[i][0]
    )

    if box[axis][0] == box[axis][1]:
        return ()

    tree = regular_index_tree if axis < 2 else backup_index_tree

    children = []

    for interval in tree[box[axis]]:
        child = box[:axis] + (interval,) + box[axis + 1:]

        if child[1][0] <= child[2][1]:
            children.append(child)

    return tuple(children)

# Cache canonical box covers for both flow networks.
from functools import lru_cache

@lru_cache(maxsize=None)
def cached_box_cover(box):
    return tuple(eligible_canonical_box_cover(*box))

#endregion

#region 7.2 Flow Map from Detailed Profile Probabilities
# Definition 7.8 and Table 2: source supports; group identical supports.
base_probability_grouped = {}

def add_base_probability(u, us, probability, support):
    key = (u, us, support)

    if key not in base_probability_grouped:
        base_probability_grouped[key] = gp.LinExpr()

    base_probability_grouped[key] += probability

for (u, us, vs), prob in profile_unmatched_us_vs_u_ANY.items():
    add_base_probability(
        u, us, 
        prob, 
        (((u, u), (us, n), (n + 1, n + 1)), ) 
    )

for (u, us), prob in profile_unmatched_us_pf.items():
    add_base_probability(
        u, us, 
        prob, 
        (((u, u), (us, us), (n + 1, n + 1)), )
    )

for (u, v ,us), prob in profile_no_backup_us_pf.items():
    add_base_probability(
        u, us, 
        prob, 
        (((u,u), (us, us), (v, v)), )
    )

for (u, v ,us), prob in profile_no_backup_us_bot.items():
    add_base_probability(
        u, us, 
        profile_no_backup_us_bot[u, v, us], 
        (((u,u), (v, v), (us, us)), )
        )

for (u, v ,us), prob in profile_no_backup_u_NO_WORSE.items():
    if us < v:
        add_base_probability(
            u, us, 
            prob, 
            (((u,u), (us, v-1), (v, v)), ((u, u), (v, v), (v, n + 1)), )
        )
    else:
        add_base_probability(
            u, us, 
            prob, 
            (((u, u), (v, v), (us, n + 1)), )
        )

for (u, v, b, us), prob in profile_backup_us_pf.items():
    add_base_probability(
        u, us,
        prob,
        (((u, u), (us, us), (v, v)), )
    )

for (u, v, b, us), prob in profile_backup_LO_b_us_bot.items():
    add_base_probability(
        u, us,
        prob,
        (((u, u), (v, v), (b, b)), )
    )

for (u, v, b, us), prob in profile_backup_HI_b_us_bot.items():
    add_base_probability(
        u, us,
        prob,
        (((u, u), (v, v), (us, us)), )
    )

for (u, v, b, us), prob in profile_backup_us_ANY_u_b.items():
    add_base_probability(
        u, us,
        prob,
        (((u, u), (b, b), (b, n + 1)), )
    )

for (u, v, b, us), prob in profile_backup_us_ANY_u_ANY.items():
    backup_upperbound = b if us > u else n + 1
    if us < v:    
        add_base_probability(
            u, us,
            prob,
            (((u, u), (us, v-1), (v, v)), ((u,u), (v, v), (v, backup_upperbound)), )
        )
    else:
        add_base_probability(
            u, us,
            prob,
            (((u,u), (v, v), (min(b, us), backup_upperbound)), )
        )

# Definition 7.10 (V7); Constraint 7.11 (C12): entry flows and source supply.
base_probability_box_inflow = {}

def add_base_probability_inflow(u, us, box, contribution):
    key = (u, us, box)

    if key not in base_probability_box_inflow:
        base_probability_box_inflow[key] = gp.LinExpr()

    base_probability_box_inflow[key] += contribution

for (u, us, support), probability_sum in base_probability_grouped.items():

    cover = [
        elligible_canonical_box
        for box in support
        for elligible_canonical_box in cached_box_cover(box)
    ]

    total = gp.LinExpr()

    for elligible_canonical_box in cover:
        flow = model.addVar()

        total += flow

        add_base_probability_inflow(u, us, elligible_canonical_box, flow)

    model.addConstr(
        total == probability_sum
    )

# Definition 7.10 (V7); Constraint 7.12 (C12): internal flows and conservation.
base_probability_box_outflow = {}

pending = list(base_probability_box_inflow)

while pending:
    u, us, box = pending.pop()
    key = (u, us, box)

    if key in base_probability_box_outflow:
        continue

    children = allowed_box_children(box)

    if not children:
        continue

    outgoing = gp.LinExpr()

    for child in children:
        flow = model.addVar()
        outgoing += flow

        add_base_probability_inflow(
            u, us, child, flow
        )

        pending.append((u, us, child))

    base_probability_box_outflow[key] = outgoing


for key, outgoing in base_probability_box_outflow.items():
    model.addConstr(
        outgoing == base_probability_box_inflow[key]
    )

# Definition 7.13 (E1): final matching expression Q as singleton inflow.
def profile_probability_with_vp_bp(u, us, vp, bp):
    box = ((u, u), (vp, vp), (bp, bp))

    return base_probability_box_inflow.get(
        (u, us, box), 0
    )

#endregion

#region 7.3 Flow Map from Victim Analysis
#region Regular and pairing source supports
# Definition 7.15 and Table 3: source-pivot supports.
victim_analysis_probability_grouped = {}

def add_victim_analysis_probability(origin, designated_vertex , probability, support):
    key = (origin, designated_vertex, support)

    if key not in victim_analysis_probability_grouped :
        victim_analysis_probability_grouped [key] = gp.LinExpr()

    victim_analysis_probability_grouped[key] += probability

for (u, v, us, vs, u2), prob in profile_no_backup_EQUAL_u_v_us_vs_u2.items():
    add_victim_analysis_probability(
        'regular',
        'vs',
        prob,
        (((vs, vs), (us, us), (u2, u2)), )
    )
    add_victim_analysis_probability(
        'regular',
        'v',
        prob,
        (((v, v), (u2, u2), (u, u)), )
    )

for (u, v, us, vs, u2), prob in profile_no_backup_LARGER_u_v_us_vs_u2.items():
    add_victim_analysis_probability(
        'regular',
        'vs',
        prob,
        (((vs, vs), (us, us), (u2, u2)), )
    )
    add_victim_analysis_probability(
        'regular',
        'u3',
        prob,
        (((vs, v), (u2, u2), (u2, u)),)
    )
    add_victim_analysis_probability(
        'regular',
        'v',
        prob,
        (((v, v), (u2 + 1, u), (u, u)),)
    )
for (u, v, us), prob in profile_no_backup_SHORT_u_v_us.items():
    add_victim_analysis_probability(
        'regular',
        'v',
        prob,
        (((v, v), (us, us), (u, u)),)
    )

for (u, v, us), prob in profile_no_backup_us_bot.items():
    add_victim_analysis_probability(
        'regular',
        'u',
        prob,
        (((u, u), (v, v), (us, us)),)
    )

for (u, v, b, us), prob in profile_backup_HI_b_us_bot.items():
    add_victim_analysis_probability(
        'regular',
        'u',
        prob,
        (((u, u), (v, v), (us, us)),)
    )

for (u, us), prob in profile_unmatched_us_pf.items():
    add_victim_analysis_probability(
        'regular',
        'u',
        prob,
        (((u, u), (us, us), (n + 1, n + 1)),)
    )

for (u, us, vs), prob in profile_unmatched_us_vs_u_ANY.items():
    add_victim_analysis_probability(
        'regular',
        'vs',
        prob,
        (((vs, vs), (us, us), (us, n)),)
    )

for (u, v, us, vs, u2, usnew), prob in P_PAIRING_EQUAL_u_v_us_vs_u2_usnew.items():
    add_victim_analysis_probability(
        ('pairing', us),
        'u2',
        prob,
        (((u2, u2), (vs, vs), (vs, v)),)
    )

for (u, v, us, vs, u2, usnew), prob in P_PAIRING_LARGER_u_v_us_vs_u2_usnew.items():
    add_victim_analysis_probability(
        ('pairing', us),
        'u2',
        prob,
        (((u2, u2), (vs, vs), (vs, v)),)
    )

    add_victim_analysis_probability(
        ('pairing', us),
        'u-2',
        prob,
        (((u2 + 1, u), (vs, v), (v, v)),)
    )

#endregion

#region Paper Section 10: Graphs with Large Odd Girth (Flow Part)
#region Odd-girth parameters and additional-pivot counts
# Section 10, Additional pivots (following Constraint 10.2): counts a_EQUAL and a_LARGER, justified by Lemma 10.1.
if odd_girth < 3 or odd_girth % 2 != 1:
    raise ValueError("odd_girth must be an odd integer >= 3")

r = (odd_girth - 1) // 2
extra_E = max(0, r - 2)
extra_L = max(0, r - 3)

#endregion

#region Additional regular sources
# Constraint 10.4 (V8/C13): additional regular sources and support boxes.
if extra_E:
    for (u, v, us, vs, u2), prob in (
        profile_no_backup_EQUAL_u_v_us_vs_u2.items()
    ):
        add_victim_analysis_probability(
            'regular',
            'odd_girth_extra_E',
            extra_E * prob,
            (((vs, v), (u2, u2), (u2, u2)),),
        )

if extra_L:
    for (u, v, us, vs, u2), prob in (
        profile_no_backup_LARGER_u_v_us_vs_u2.items()
    ):
        add_victim_analysis_probability(
            'regular',
            'odd_girth_extra_L',
            extra_L * prob,
            (((vs, v), (u2, u), (u2, u)),),
        )

#endregion

#region Additional pairing sources
# Constraint 10.4 (V8/C13): additional pairing sources and support boxes.
# Sources retain their earlier-rank graph index us.
if extra_E:
    for (u, v, us, vs, u2, usnew), prob in (
        P_PAIRING_EQUAL_u_v_us_vs_u2_usnew.items()
    ):
        add_victim_analysis_probability(
            ('pairing', us),
            'odd_girth_extra_E',
            extra_E * prob,
            (((u2, u2), (vs, v), (vs, v)),),
        )

if extra_L:
    for (u, v, us, vs, u2, usnew), prob in (
        P_PAIRING_LARGER_u_v_us_vs_u2_usnew.items()
    ):
        add_victim_analysis_probability(
            ('pairing', us),
            'odd_girth_extra_L',
            extra_L * prob,
            (((u2, u), (vs, v), (vs, v)),),
        )

#endregion

#endregion

#region Victim flows and final-profile bounds
# Definition 7.17 (V8); Constraint 7.18 (C13), extended by 10.4: entry supply.
victim_analysis_probability_box_inflow = {}

def add_victim_analysis_probability_inflow(origin, box, contribution):
    key = (origin, box)

    if key not in victim_analysis_probability_box_inflow:
        victim_analysis_probability_box_inflow[key] = gp.LinExpr()

    victim_analysis_probability_box_inflow[key] += contribution

for (origin, designated_vertex, support), probability_sum in victim_analysis_probability_grouped.items():

    cover = [
        canonical_box
        for box in support
        for canonical_box in cached_box_cover(box)
    ]

    total = gp.LinExpr()

    for canonical_box in cover:
        flow = model.addVar()

        total += flow

        add_victim_analysis_probability_inflow(
            origin, canonical_box, flow
        )

    model.addConstr(total == probability_sum)


# Definition 7.17 (V8); Constraint 7.19 (C13), extended by 10.5: internal conservation.
victim_analysis_probability_box_outflow = {}

pending = list(victim_analysis_probability_box_inflow)

while pending:
    origin, box = pending.pop()
    key = (origin, box)

    if key in victim_analysis_probability_box_outflow:
        continue

    children = allowed_box_children(box)

    if not children:
        continue

    outgoing = gp.LinExpr()

    for child in children:
        flow = model.addVar()
        outgoing += flow

        add_victim_analysis_probability_inflow(
            origin, child, flow
        )

        pending.append((origin, child))

    victim_analysis_probability_box_outflow[key] = outgoing


for key, outgoing in victim_analysis_probability_box_outflow.items():
    model.addConstr(
        outgoing == victim_analysis_probability_box_inflow[key]
    )

# Victim singleton inflows and pairing graph indices.
def profile_probability_with_vp_bp_from_victim_analysis(origin, u, vp, bp):
    box = ((u, u), (vp, vp), (bp, bp))

    return victim_analysis_probability_box_inflow.get(
        (origin, box), 0
    )

us_candidate_cells = sorted({
    us for us, usnew in pairing_indices
})

# Constraint 7.20 (C14), extended by 10.6: regular + pairing flow <= final profile.
# The omitted bp=n+1 case follows from Table 2 and the C15 equalities.
for (u, vp, bp), prob in P_u_vp_bp.items():
    regular = profile_probability_with_vp_bp_from_victim_analysis(
        'regular', u, vp, bp
    )

    for us in us_candidate_cells:
        pairing = profile_probability_with_vp_bp_from_victim_analysis(
            ('pairing', us), u, vp, bp
        )

        model.addConstr(
            regular + pairing <= prob
        )

#endregion

#endregion

#region 7.4 Final Match Expression Constraints
# Constraint 7.21 (C15): both final-profile marginals of Q.
for (u, vp, bp), prob in P_u_vp_bp.items():
    total = gp.quicksum(
        profile_probability_with_vp_bp(u, us, vp, bp)
        for us in range(1, n + 1)
            )
    model.addConstr(
        total == prob
    )

for (u, vp, us), prob in P_u_vp_us.items():
    total = gp.quicksum(
        profile_probability_with_vp_bp(u, us, vp, bp)
        for bp in range(vp, n + 2)
                )
    model.addConstr(
        total == prob
    )

# Constraint 7.22 (C16): refined SHORT probability <= Q.
for (u, v, us, bs), refined_prob in profile_no_backup_SHORT_u_v_us_bs.items():
    model.addConstr(
        refined_prob <= profile_probability_with_vp_bp(us, u, v, bs)
    )

# Constraint 7.23 (C17): low backup or low-ranked match of u*.
for u in range(1, n + 1):
    for us in range(1, n + 1):
        LHS = gp.LinExpr()

        LHS += gp.quicksum(
            profile_probability_with_vp_bp(u, us, vp, bp)
            for bp in range(1, us + 1)
            for vp in range(1, bp + 1)
        )

        LHS += gp.quicksum(
            P_u_vp_us[us, vs, u]
            for vs in range(1, u + 1)
        )

        model.addConstr(
            LHS >= 1 / (n * n)
        )

#endregion

#endregion

#region 8. Monotonicity Constraints
#region 8.1 Demoting v
# Constraint 8.1 (C18): coarse backup monotonicity and its refinement.
for u in range(1, n + 1):
    for v in range(1, n + 1):
        for b in range(v+2, n + 1):
            model.addConstr(coarse_profile_backup[u, v , b] <= coarse_profile_backup[u, v + 1, b])
            for us in range(1, u + 1):
                LHS = 1 / n * coarse_profile_backup[u, v, b] - profile_backup_us_ANY_u_b[u, v, b, us] 
                RHS = 1 / n * coarse_profile_backup[u, v + 1, b] - profile_backup_us_ANY_u_b[u, v + 1, b, us]
                model.addConstr(LHS <= RHS)

# Constraint 8.2 (C19): demoting v beyond u after a SHORT loss.
for u in range(1, n + 1):
    for v in range(1, u + 1):
        for us in range(1, u + 1):
            total_prob = profile_no_backup_SHORT_u_v_us[u, v, us]
            total_prob -= gp.quicksum(
                profile_backup_LO_b_us_bot[us, v, b, u]
                for b in range(v, u + 1)
            )
            for j in range(u + 1, n + 1):
                model.addConstr(total_prob <= profile_no_backup_us_pf[u, j, us])

# Definition 8.6 (C21): v-transition variables.
DELTA_profile_no_backup_EQUAL_u_v_us_vs_u2 = {}
DELTA_profile_no_backup_LARGER_u_v_us_vs_u2 = {}
DELTA_profile_no_backup_SHORT_u_v_us_bs = {}
DELTA_profile_no_backup_us_vs_OBLIVIOUS = {}
DELTA_profile_no_backup_us_bot = {}

# Definition 8.6 (C21): transition difference lower bounds.
for u, v ,us, vs, u2 in profile_no_backup_EQUAL_u_v_us_vs_u2.keys():
    if v == n:
        continue
    var = model.addVar()
    DELTA_profile_no_backup_EQUAL_u_v_us_vs_u2[u, v ,us, vs, u2] = var

    model.addConstr(
        var >= profile_no_backup_EQUAL_u_v_us_vs_u2[u, v ,us, vs, u2] - profile_no_backup_EQUAL_u_v_us_vs_u2[u, v + 1, us, vs, u2]
    )

for u, v ,us, vs, u2 in profile_no_backup_LARGER_u_v_us_vs_u2.keys():
    if v == n:
        continue
    var = model.addVar()
    DELTA_profile_no_backup_LARGER_u_v_us_vs_u2[u, v ,us, vs, u2] = var

    model.addConstr(
        var >= profile_no_backup_LARGER_u_v_us_vs_u2[u, v ,us, vs, u2] - profile_no_backup_LARGER_u_v_us_vs_u2[u, v + 1, us, vs, u2]
    )

for u, v, us, bs in profile_no_backup_SHORT_u_v_us_bs.keys():
    if v >= min(n, bs):
        continue
    var = model.addVar()
    DELTA_profile_no_backup_SHORT_u_v_us_bs[u, v ,us, bs] = var

    model.addConstr(
        var >= profile_no_backup_SHORT_u_v_us_bs[u, v ,us, bs] - profile_no_backup_SHORT_u_v_us_bs[u, v + 1, us, bs]
    )

for u, v, us, vs in profile_no_backup_us_vs_OBLIVIOUS.keys():
    if us == v + 1 and vs > u:
        var = model.addVar()
        DELTA_profile_no_backup_us_vs_OBLIVIOUS[u, v ,us, vs] = var

        model.addConstr(
            var >= profile_no_backup_us_vs_OBLIVIOUS[u, v ,us, vs] - profile_no_backup_us_vs_OBLIVIOUS[u, v + 1, us, vs]
        )

for u, v, us in profile_no_backup_us_bot.keys():
    if us == v + 1: 
        var = model.addVar()
        DELTA_profile_no_backup_us_bot[u, v ,us] = var

        model.addConstr(
            var >= profile_no_backup_us_bot[u, v ,us] - profile_no_backup_us_bot[u, v + 1, us]
        )

# Definition 8.5 (C20, Qboth expanded inline); Constraint 8.7 (C22).
for u, v, us, vs in profile_no_backup_us_vs_OBLIVIOUS.keys():
    if v == n:
        continue  
    if vs <= u or us > v + 1:

        LHS = gp.LinExpr()
        RHS = gp.LinExpr()

        LHS += profile_no_backup_us_vs_OBLIVIOUS[u, v, us, vs]
        RHS += profile_no_backup_us_vs_OBLIVIOUS[u, v + 1, us, vs]

        if vs == v:
            LHS -= profile_no_backup_SHORT_u_v_us.get((u, v, us), 0)
        if vs == v + 1:
            RHS -= profile_no_backup_SHORT_u_v_us.get((u, v + 1, us), 0)

        LHS -= gp.quicksum(
            profile_no_backup_EQUAL_u_v_us_vs_u2.get((u, v, us, vs, u2), 0) + profile_no_backup_LARGER_u_v_us_vs_u2.get((u, v, us, vs, u2), 0)
            for u2 in range(us, u + 1)
        )
        RHS -= gp.quicksum(
            profile_no_backup_EQUAL_u_v_us_vs_u2.get((u, v + 1, us, vs, u2), 0) + profile_no_backup_LARGER_u_v_us_vs_u2.get((u, v + 1, us, vs, u2), 0)
            for u2 in range(us, u + 1)
        )


        LHS += gp.quicksum(
            DELTA_profile_no_backup_EQUAL_u_v_us_vs_u2.get((u, v, us, vs, u2), 0,) + DELTA_profile_no_backup_LARGER_u_v_us_vs_u2.get((u, v, us, vs, u2), 0)
            for u2 in range(us, u + 1)
        )
            
        if vs > v:
            LHS += DELTA_profile_no_backup_SHORT_u_v_us_bs.get((u, v, us, vs), 0)
        if vs == v:
            LHS += profile_no_backup_SHORT_u_v_us_bs.get((u, v, us, v), 0)
        if vs == u and us == v + 1:
            LHS +=DELTA_profile_no_backup_us_bot.get((u, v, us), 0)
            LHS += gp.quicksum(
                DELTA_profile_no_backup_us_vs_OBLIVIOUS.get((u, v, us, bs), 0)
                for bs in range(u + 1, n + 1)
            )
        if vs == u and us == v :
            LHS += profile_no_backup_us_bot.get((u, v, us), 0)
            LHS += gp.quicksum(
                profile_no_backup_us_vs_OBLIVIOUS.get((u, v, us, bs), 0)
                for bs in range(u + 1, n + 1)
            )

        model.addConstr(LHS <= RHS)

#endregion

#region 8.2 Demoting u
# Definition 8.8 (C23): u-transition variables and difference lower bounds.
DELTA_U_profile_no_backup_EQUAL_u_v_us_vs_u2 = {}
DELTA_U_profile_no_backup_LARGER_u_v_us_vs_u2 = {}
DELTA_U_profile_no_backup_SHORT_u_v_us = {}

for u, v ,us, vs, u2 in profile_no_backup_EQUAL_u_v_us_vs_u2.keys():
    if u == n:
        continue
    var = model.addVar()
    DELTA_U_profile_no_backup_EQUAL_u_v_us_vs_u2[u, v ,us, vs, u2] = var
    later_prob = profile_no_backup_EQUAL_u_v_us_vs_u2[u + 1, v, us, vs, u2]

    model.addConstr(
        var >= profile_no_backup_EQUAL_u_v_us_vs_u2[u, v ,us, vs, u2] - later_prob
    )

for u, v ,us, vs, u2 in profile_no_backup_LARGER_u_v_us_vs_u2.keys():
    if u == n:
        continue
    var = model.addVar()
    DELTA_U_profile_no_backup_LARGER_u_v_us_vs_u2[u, v ,us, vs, u2] = var
    later_prob = profile_no_backup_LARGER_u_v_us_vs_u2[u + 1, v, us, vs, u2]
    model.addConstr(
        var >= profile_no_backup_LARGER_u_v_us_vs_u2[u, v ,us, vs, u2] - later_prob
    )

for u, v, us in profile_no_backup_SHORT_u_v_us.keys():
    if u == n:
        continue
    var = model.addVar()
    DELTA_U_profile_no_backup_SHORT_u_v_us[u, v ,us] = var
    later_prob = profile_no_backup_SHORT_u_v_us[u + 1, v, us]
    model.addConstr(
        var >= profile_no_backup_SHORT_u_v_us[u, v ,us] - later_prob
    )

# Constraint 8.9 (C24): unmatched-profile monotonicity with transition terms.
for u, us, vs in profile_unmatched_us_vs_u_bot.keys():
    if u == n or us > u:
        continue

    LHS = gp.LinExpr()
    LHS += profile_unmatched_us_vs_u_bot[u, us, vs]

    LHS += gp.quicksum(
        DELTA_U_profile_no_backup_EQUAL_u_v_us_vs_u2[u, v, us, vs, u2]
        for v in range(vs, n + 1)
        for u2 in range(us, u + 1)
    )

    LHS += gp.quicksum(
        DELTA_U_profile_no_backup_LARGER_u_v_us_vs_u2[u, v, us, vs, u2]
        for v in range(vs, n + 1)
        for u2 in range(us, u)
    )

    LHS += DELTA_U_profile_no_backup_SHORT_u_v_us[u, vs, us]

    RHS = profile_unmatched_us_vs_u_bot[u + 1, us, vs]

    model.addConstr(LHS <= RHS)

# Constraint 8.10 (C25): unmatched-profile monotonicity for us > u.
for u, us, vs in profile_unmatched_us_vs_u_bot.keys():
    if u == n or us <= u:
        continue
    model.addConstr(
        profile_unmatched_us_vs_u_bot[u, us, vs] <= profile_unmatched_us_vs_u_bot[u + 1, us, vs]
    )

#endregion

#region 8.3 Demoting u*
# Constraint 8.11 (C26): cumulative OBLIVIOUS-profile monotonicity.
for u in range(1, n + 1):
    for v in range(1, n + 1):
        for us in range(1, n):
            for k in range(1, n + 1):
                if us < v and k > u:
                    continue
                LHS = gp.quicksum(
                    profile_no_backup_us_vs_OBLIVIOUS[u, v, us, vs]
                    for vs in range(1, k + 1)
                )
                RHS = gp.quicksum(
                    profile_no_backup_us_vs_OBLIVIOUS[u, v, us+1, vs]
                    for vs in range(1, k + 1)
                )
                model.addConstr(LHS>= RHS)

#endregion

#region 8.4 Demoting v'
# Constraint 8.12 (C27): final-profile monotonicity with excluded events.
for u in range(1, n + 1):
    for us in range(1, n + 1):
        for vp in range(1, n + 1):
            for bp in range(vp + 1, n + 2):
                if bp <= vp + 1:
                    continue
                LHS = gp.LinExpr()
                RHS = gp.LinExpr()

                LHS += profile_probability_with_vp_bp(u, us, vp, bp)
                RHS += profile_probability_with_vp_bp(u, us, vp + 1, bp)

                if bp == us:
                    LHS -= profile_no_backup_us_bot[u, vp, us]
                    LHS -= gp.quicksum(
                        profile_backup_HI_b_us_bot[u, vp, b, us]
                        for b in range(us, n + 1)
                    )
                    RHS -= profile_no_backup_us_bot[u, vp + 1, us]
                    RHS -= gp.quicksum(
                        profile_backup_HI_b_us_bot[u, vp + 1, b, us]
                        for b in range(us, n + 1) 
                    )
                if bp <= us:
                    LHS -= profile_backup_LO_b_us_bot[u, vp, bp, us]
                    RHS -= profile_backup_LO_b_us_bot[u, vp + 1, bp, us]

                if bp == n + 1 and vp == us:
                    LHS -= profile_unmatched_us_pf[u, us]
                if bp == n + 1 and vp + 1 == us:
                    RHS -= profile_unmatched_us_pf[u, us]

                if vp == us and us <= bp and bp <= n:
                    LHS -= profile_no_backup_us_pf[u, bp, us]
                    LHS -= gp.quicksum(
                        profile_backup_us_pf[u, bp, b, us]
                        for b in range(bp, n + 1)
                    )

                if vp + 1 == us and us <= bp and bp <= n:
                    RHS -= profile_no_backup_us_pf[u, bp, us]
                    RHS -= gp.quicksum(
                        profile_backup_us_pf[u, bp, b, us]
                        for b in range(bp, n + 1)
                    )

                model.addConstr(LHS <= RHS)

#endregion

#region 8.5 Same Interval Monotonicity Constraints
# Constraint 8.13 (C28): half-probability OBLIVIOUS bound.
for u in range(1, n + 1):
    for us in range(1, n + 1):
        LHS = gp.quicksum(
            profile_no_backup_us_vs_OBLIVIOUS[u, us, us, j]
            for j in range(1, u + 1)
        )
        model.addConstr( 
            LHS >= 1 / (2 * n) * coarse_profile_no_backup[u, us] 
        )

# Constraint 8.14 (C29): combined same-interval lower bound.
for u in range(1, n + 1):
    for us in range(1, n + 1):
        LHS = gp.quicksum(
            P_u_vp_us[us, j, u]
            for j in range(1, u + 1)
        )

        RHS = gp.LinExpr()
        RHS += 1 / n * coarse_profile_unmatched[u]
        RHS += gp.quicksum(
            profile_no_backup_us_vs_OBLIVIOUS[u, v, us, vs]
            for v in range(1, n + 1)
            for vs in range(1, u + 1)
        )
        RHS += 1 / n * gp.quicksum(
            coarse_profile_backup[u, v, b]
            for v in range(us + 1, n + 1)
            for b in range(v, n + 1)
        )
        RHS += 1 / (2 * n) * gp.quicksum(
            coarse_profile_backup[u, us, b]
            for b in range(us + 1, n + 1)
        )

        model.addConstr( 
            LHS >= RHS
        )

# Constraint 8.15 (C30): same-cell worsening probability <= 1/(2n^2).
for i in range(1, n + 1):
    total=gp.LinExpr()
    total += gp.quicksum(
        profile_no_backup_WORSE_u_ANY_us_vs[i, i, t]
        for t in range(1, i + 1)
    )

    total += gp.quicksum(
        profile_backup_us_ANY_u_b[i, v, b, i]
        for v in range(1, n + 1)
        for b in range(v, n + 1)
    )

    model.addConstr(
        total <= 1 / (2 * n * n),
    )

# Constraint 8.16 (C31): SHORT-path monotonicity up to the cell of u.
for u in range(1, n + 1):
    LHS = gp.quicksum(
        profile_no_backup_SHORT_u_v_us[u, v, u]
        for v in range(1, u + 1)
    )
    for us in range(1, u):
        RHS = 1 / 8 * gp.quicksum(
            profile_no_backup_SHORT_u_v_us[u, v, us]
            for v in range(1, u + 1)
        ) - 1 / (128 * n * n)

        model.addConstr(
            LHS >= RHS
        )

#endregion

#endregion

#region 9. Symmetrical Constraints
#region Symmetry and reciprocal bounds
# Constraint 9.1 (C32): symmetry of final matched-rank probabilities.
for u in range(1, n + 1):
    for v in range(1, n + 1):
        LHS = gp.quicksum(
            P_u_vp_us[u, v, j]
            for j in range(1, n + 1)
        )
        RHS = gp.quicksum(
            P_u_vp_us[v, u, j]
            for j in range(1, n + 1)
        )
        model.addConstr(LHS == RHS)

# Constraint 9.2 (C33): reciprocal descriptions of an unmatched vertex.
for u in range(1, n + 1):
    for us in range(1, n + 1):
        for vs in range(1, u + 1):
            LHS=gp.LinExpr()
            RHS=gp.LinExpr()

            LHS += profile_unmatched_us_vs_u_bot[u, us, vs]
            if us<= u:
                LHS += profile_no_backup_WORSE_u_ANY_us_vs[u, us, vs]

            RHS += profile_no_backup_us_bot[us, vs, u]
            for bs in range(vs, u+1):
                RHS += profile_backup_LO_b_us_bot[us, vs, bs, u]
            for bs in range(u, n+1):
                RHS += profile_backup_HI_b_us_bot[us, vs, bs, u]

            model.addConstr(LHS == RHS)

# Constraint 9.3 (C34): lower bound from the optimum-partner perspective.
for u in range(1, n + 1):
    for us in range(1, n + 1):
        for vs in range (1, n + 1):
            total_prob = gp.LinExpr()
            total_prob += gp.quicksum(
                profile_no_backup_us_vs_OBLIVIOUS[u, v, us, vs]
                for v in range(1, n + 1)
            )
            if vs<= u:
                total_prob += profile_unmatched_us_vs_u_bot[u ,us, vs]
                total_prob += profile_unmatched_us_vs_u_ANY[u, us, vs]
            if vs==u:
                total_prob += profile_unmatched_us_pf[u, us]
                total_prob += gp.quicksum(
                    profile_backup_us_pf[u, v, b, us]
                    for v in range(us, n + 1)
                    for b in range(v, n + 1)
                )
            model.addConstr(P_u_vp_us[us, vs, u] >= total_prob)

#endregion

#region Earlier-matching probabilities
# Definition 9.4 (V9): probability that the first vertex matches earlier.
P_EARLIER_us_vs_u= {
    (us, vs, u) : model.addVar(ub = 1 / (n * n)) 
    for us in range(1, n + 1)
    for vs in range(1, n + 1)
    for u in range(vs, n + 1)
}

# Constraint 9.5 (C35): elementary earlier-matching lower bound.
for us in range(1, n + 1):
    for vs in range(1, n + 1):
        for u in range(vs, n + 1):
            LHS=gp.LinExpr()
            LHS += profile_unmatched_us_vs_u_bot[u, us, vs] + profile_unmatched_us_vs_u_ANY[u, us, vs]
            if us <= u:
                LHS += profile_no_backup_WORSE_u_ANY_us_vs[u, us ,vs]    
            model.addConstr(LHS <= P_EARLIER_us_vs_u[us, vs, u])  

# Constraint 9.6 (C36): cumulative earlier-matching bound including backups.
for us in range(1, n + 1):
    for u in range(1, n + 1):
        for k in range(1, u + 1):
            LHS=gp.LinExpr()
            RHS=gp.LinExpr()
            LHS += gp.quicksum(P_EARLIER_us_vs_u[us, vs, u] for vs in range(1, k + 1))
            RHS += gp.quicksum(
                profile_unmatched_us_vs_u_bot[u, us, vs] +  profile_unmatched_us_vs_u_ANY[u, us, vs] + profile_no_backup_WORSE_u_ANY_us_vs.get((u, us, vs), 0) 
                for vs in range(1, k + 1)
            )
            RHS += gp.quicksum(
                profile_backup_us_ANY_u_b.get((u, v, b, us), 0)
                for v in range(1, n + 1)
                if min(u, v) <= k
                for b in range(v, n + 1)
            )
            if k == u:
                RHS += gp.quicksum(
                    profile_no_backup_u_NO_WORSE[u, v, us]
                    for v in range(us + 1, n + 1)
                )
                RHS += gp.quicksum(
                    profile_backup_us_ANY_u_ANY[u, v, b, us]
                    for v in range(us + 1, n + 1)
                    for b in range(v , n + 1)
                )
            model.addConstr(LHS >= RHS)

#endregion

#region Combined event bounds
# Definition 9.7 (Qpf, expanded inline); Constraint 9.8 (C37).
for u in range(1, n + 1):
    for vp in range (1, n + 1):
        for us in range(vp, n + 1):
            RHS = gp.LinExpr()
            RHS += P_EARLIER_us_vs_u[u, vp , us]
            if vp == us:
                RHS += profile_unmatched_us_pf[u, us]
                RHS += gp.quicksum(
                    profile_no_backup_us_pf[u, v, us]
                    for v in range(us, n + 1)
                )
                RHS += gp.quicksum(
                    profile_backup_us_pf[u, v, b, us]
                    for v in range(us, n + 1)
                    for b in range(v, n + 1)
                    )
            model.addConstr(P_u_vp_us[u, vp, us] >= RHS)

# Definition 9.7 (Qpf, expanded inline); Constraint 9.9 (C38): event partition.
for u in range(1, n + 1):
    for us in range(1, n + 1):
        total_prob=gp.LinExpr()
        total_prob += gp.quicksum(
            P_EARLIER_us_vs_u[us, j, u] 
            for j in range(1, u + 1)
        )
        total_prob += gp.quicksum(
            P_EARLIER_us_vs_u[u, j, us] 
            for j in range(1, us + 1)
        )
        total_prob += profile_unmatched_us_pf[u, us]
        total_prob += gp.quicksum(
            profile_no_backup_us_pf[u, v, us]
            for v in range(us, n + 1)
        )
        total_prob += gp.quicksum(
            profile_backup_us_pf[u, v, b, us]
            for v in range(us, n + 1)
            for b in range(v, n + 1)
            )
        model.addConstr(total_prob == 1 / (n * n))

#endregion

#region Interval blocker bound
# Constraint 9.10 (C39): interval blocker/victim bound.
for v in range(1, n + 1):
    for L in range(1, n + 1):
        for R in range(L, n + 1):
            LHS = gp.LinExpr()
            RHS = gp.LinExpr()

            LHS += gp.quicksum(
                profile_no_backup_EQUAL_u_v_us_vs_u2[u, v, us, vs, u2]
                for u in range(L, R + 1)
                for us in range(1, u + 1)
                for vs in range(1, min(u, v) + 1)
                for u2 in range(max(us, L), u + 1)
            )
            LHS += gp.quicksum(
                profile_no_backup_LARGER_u_v_us_vs_u2[u, v, us, vs, u2]
                for u in range(L, R + 1)
                for us in range(1, u)
                for vs in range(1, min(u, v) + 1)
                for u2 in range(max(us, L - 1), u)
            )
            for z in range(L, R + 1):
                RHS += gp.quicksum(
                    P_u_vp_us[z, v, zs]
                    for zs in range(1, n + 1)
                )
                RHS -=gp.quicksum(
                    profile_no_backup_WORSE_u_ANY_us_vs[zs, z, v]
                    for zs in range(max(z, v), n + 1)
                )
                RHS -= gp.quicksum(
                    profile_no_backup_us_bot[v, z, j]
                    for j in range(z, n + 1)
                )
                RHS -= gp.quicksum(
                    profile_backup_HI_b_us_bot[v, z, b, j]
                    for j in range(z, n + 1)
                    for b in range(j, n + 1)
                )
                RHS -= gp.quicksum(
                    profile_unmatched_us_vs_u_ANY[j, z, v]
                    for j in range(v, n + 1)
                )
                RHS -= profile_unmatched_us_pf[v, z]
                RHS -= gp.quicksum(
                    profile_no_backup_us_pf[v, j, z]
                    for j in range(R + 1, n + 1)
                )
                RHS -= gp.quicksum(
                    profile_backup_us_pf[v, j, b, z]
                        for j in range(R + 1, n + 1)
                        for b in range(j, n + 1)
                )

            model.addConstr(LHS <= RHS)

#endregion

#endregion

#region 10. Odd-girth Constraints
# Flow-related constraints from Section 10 are integrated into Section 7.3 above.

# Constraint 10.2 (C40): SHORT events have zero upper bound for odd_girth >= 5.
if odd_girth >= 5:
    for prob in profile_no_backup_SHORT_u_v_us.values():
        prob.UB = 0.0

# Constraint 10.9 (C41): tau-boundary-crossing constraints.
boundary_path_length = (odd_girth - 1) // 2
for tau in range(1, n + 1):
    boundary_crossing_probability = gp.LinExpr()

    # u retains a backup match, but crosses the boundary: v <= tau < b.
    boundary_crossing_probability += gp.quicksum(
        prob
        for (u, v, b, us), prob in profile_backup_us_ANY_u_b.items()
        if v <= tau < b
    )

    model.addConstr(
        boundary_path_length * boundary_crossing_probability <= tau / n
    )
#endregion

#region Solve and Output
model.update()

model.optimize()

print(f"n: {n}")
print(f"Runtime: {model.Runtime:.2f} seconds")

if model.SolCount > 0:
    print(f"Objective: {model.ObjVal:.12f}")
else:
    print("No solution available.")
#endregion

