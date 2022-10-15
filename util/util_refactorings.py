import numpy as np
from more_itertools import unique_everseen
import pandas as pd
from collections import defaultdict
import copy


def read(pos, tree):
    count = []
    while (pos > 0):
        count += tree[pos]
        pos -= (pos & -pos)
    return count


def update(pos, MAX, edge, tree):
    while (pos <= MAX):
        tree[pos].append(edge)
        pos += (pos & -pos)


def _find_crossings(n, m, e):
    """
    Method to find crossings on a bipartite graph

    n: num nodes on left
    m: num nodes on right
    e: edges -> tuples of (node_idx_left, node_idx_right)
    """
    tree = defaultdict(list)
    k = len(e)
    e = sorted(e)
    res = {}
    for i in range(k):
        r_m = read(m, tree)
        r_e = read(e[i][1], tree)
        c = set(r_m) - set(r_e)
        res[e[i]] = c
        update(e[i][1], m, e[i], tree)
    ##
    return res, tree


def remove_one_crossing(dict_r):
    # 1. select the edge with the most crossings
    max_crossing_val = len(max(dict_r.values(), key=lambda x: len(x)))
    chosen_keys = filter(lambda x: len(x[1]) == max_crossing_val, dict_r.items())
    chosen_keys = list(dict(chosen_keys).keys())

    # 1a. if only one crossing, continue
    if len(chosen_keys) == 1:
        key_to_remove = chosen_keys[0]

    # 1b. if there's multiple keys with the same number of crossings, take the ones moves the farthest.
    else:
        max_move = max(map(lambda x: abs(x[1] - x[0]), chosen_keys))
        chosen_keys = list(filter(lambda x: abs(x[1] - x[0]) == max_move, chosen_keys))
        if len(chosen_keys) == 1:
            key_to_remove = chosen_keys[0]

        # 1c. If there are multiple keys that move the same distance, take the ones that move up.
        # 1d. If there are multiple of these keys, just take the first
        else:
            moves_up = list(filter(lambda x: x[1] - x[0] < 0, chosen_keys))
            if len(moves_up) > 0:
                key_to_remove = moves_up[0]
            else:
                key_to_remove = chosen_keys[0]

    dict_r.pop(key_to_remove)
    for key, crossings in dict_r.items():
        if key_to_remove in crossings:
            dict_r[key].remove(key_to_remove)

    return key_to_remove, dict_r


def identify_refactor_edges(crossings_dict):
    r_copy = copy.deepcopy(crossings_dict)
    removed_crossings = []
    while any(r_copy.values()):
        removed_crossing, r_copy = remove_one_crossing(r_copy)
        removed_crossings.append(removed_crossing)
    return removed_crossings


def symmetrize_crossings(crossings):
    """
    Make sure that the crossings list is symmetrical.

    :param crossings:
    :return:
    """
    for start_edge, edge_crossing_set in crossings.items():
        for end_edge in list(edge_crossing_set):
            if start_edge not in crossings[end_edge]:
                crossings[end_edge].add(start_edge)
    return crossings


def find_refactors_for_doc(one_doc=None, sents_old=None, sents_new=None):
    """
    Method to find refactorings (i.e. whether pairs of sentences cross each other in a bipartite graph)

    params:
    * one_doc: a dataframe containing columns: ['sent_idx_x', 'sent_idx_y']
    OR
    * sents_old: list (or pd.Series) of sentences from the old version
    * sents_new: list (or pd.Series) of sentences from the new version

    returns:
    * num_crossings: the number of sentences that have been refactored,
        i.e. the number of edges in a bipartite graph of sentences that cross each other.
    """
    # make it not zero-indexed, for bitwise addition
    correct_zero = lambda x: x + 1

    # drop additions/deletions (these don't affect refactorings)
    if one_doc is not None:
        one_doc = one_doc.sort_values(['sent_idx_x', 'sent_idx_y'])
        one_doc = one_doc.loc[lambda df: df[['sent_idx_x', 'sent_idx_y']].notnull().all(axis=1)]
        e_pre_map = one_doc[['sent_idx_x', 'sent_idx_y']].astype(int).apply(lambda x: tuple(x), axis=1)

        sents_old_idx_pre_map = sorted(list(set(list(map(lambda x: x[0], e_pre_map)))))
        sents_new_idx_pre_map = sorted(list(set(list(map(lambda x: x[1], e_pre_map)))))

        # map missing indices (the result of dropping additions/deletions) to a compressed set.
        sents_old_map = {v: correct_zero(k) for k, v in enumerate(unique_everseen(sents_old_idx_pre_map))}
        sents_new_map = {v: correct_zero(k) for k, v in enumerate(unique_everseen(sents_new_idx_pre_map))}

        #
        e = []
        for e_i in e_pre_map:
            e_i_post = (sents_old_map[e_i[0]], sents_new_map[e_i[1]])
            e.append(e_i_post)

    if sents_old is not None and sents_new is not None:
        sents_old = list(filter(pd.notnull, sents_old))
        sents_new = list(filter(pd.notnull, sents_new))
        sents_old = list(map(int, sents_old))
        sents_new = list(map(int, sents_new))

        # map missing indices (the result of dropping additions/deletions) to a compressed set.
        sents_old_map = {v: correct_zero(k) for k, v in enumerate(unique_everseen(sents_old))}
        sents_new_map = {v: correct_zero(k) for k, v in enumerate(unique_everseen(sents_new))}

        sents_old = list(map(sents_old_map.get, sents_old))
        sents_new = list(map(sents_new_map.get, sents_new))
        e = list(zip(sents_old, sents_new))

    # prepare input to function
    n = len(sents_old_map)
    m = len(sents_new_map)

    # calculate and return
    crossings, tree = _find_crossings(n, m, e)
    crossings = symmetrize_crossings(crossings)
    refactors = identify_refactor_edges(crossings)
    if len(refactors) > 0:
        sents_old_map_r = {v: k for k, v in sents_old_map.items()}
        sents_new_map_r = {v: k for k, v in sents_new_map.items()}
        refactors = list(map(lambda x: (sents_old_map_r[x[0]], sents_new_map_r[x[1]]), refactors))
    return refactors
