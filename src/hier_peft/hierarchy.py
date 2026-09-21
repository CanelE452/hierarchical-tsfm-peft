from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HierarchyGraph:
    ids: tuple[str, ...]
    bottom_ids: tuple[str, ...]
    supports: tuple[frozenset[int], ...]
    parents: dict[int, tuple[int, ...]]
    children: dict[int, tuple[int, ...]]
    root_indices: tuple[int, ...]
    leaf_indices: tuple[int, ...]
    duplicate_supports: dict[tuple[int, ...], tuple[int, ...]]
    isolated_indices: tuple[int, ...]
    nonroot_without_parent: tuple[int, ...]
    nonleaf_without_child: tuple[int, ...]
    edge_count: int
    structure_sha256: str

    @property
    def parents_by_id(self) -> dict[str, list[str]]:
        return {
            self.ids[i]: [self.ids[parent] for parent in parents]
            for i, parents in self.parents.items()
        }

    @property
    def children_by_id(self) -> dict[str, list[str]]:
        return {
            self.ids[i]: [self.ids[child] for child in children]
            for i, children in self.children.items()
        }


def _as_summing_array(S_df: pd.DataFrame, tolerance: float) -> np.ndarray:
    try:
        values = S_df.to_numpy(dtype=np.float64)
    except ValueError as exc:
        raise ValueError("S_df must be numeric") from exc
    if values.ndim != 2:
        raise ValueError("S_df must be a two-dimensional matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("S_df contains missing or nonfinite values")
    distance_to_binary = np.minimum(np.abs(values), np.abs(values - 1.0))
    max_binary_distance = float(distance_to_binary.max(initial=0.0))
    if max_binary_distance > tolerance:
        raise ValueError(
            f"S_df must be binary within tolerance; max distance {max_binary_distance}"
        )
    return (np.abs(values) > tolerance).astype(np.int8)


def support_sets_from_S(S_df: pd.DataFrame, tolerance: float = 0.0) -> tuple[frozenset[int], ...]:
    """Return bottom-column support sets for every hierarchy row in S_df."""
    S = _as_summing_array(S_df, tolerance)
    supports: list[frozenset[int]] = []
    for row in S:
        support = frozenset(np.flatnonzero(row).astype(int).tolist())
        if not support:
            raise ValueError("Every S_df row must support at least one bottom series")
        supports.append(support)
    return tuple(supports)


def build_support_inclusion_graph(
    S_df: pd.DataFrame,
    *,
    tolerance: float = 0.0,
) -> HierarchyGraph:
    """Build a direct parent/child graph from summing-matrix support inclusion only.

    A direct parent is a minimal strict support superset. Series names are carried
    through for reporting only; they are never parsed to infer graph edges.
    """
    if S_df.index.has_duplicates:
        raise ValueError("S_df index must be unique")
    if S_df.columns.has_duplicates:
        raise ValueError("S_df columns must be unique")

    ids = tuple(str(value) for value in S_df.index)
    bottom_ids = tuple(str(value) for value in S_df.columns)
    if len(set(ids)) != len(ids):
        raise ValueError("S_df index values collide after string conversion")
    if len(set(bottom_ids)) != len(bottom_ids):
        raise ValueError("S_df column values collide after string conversion")

    supports = support_sets_from_S(S_df, tolerance)
    full_support = frozenset(range(len(bottom_ids)))

    support_to_nodes: dict[frozenset[int], list[int]] = {}
    for idx, support in enumerate(supports):
        support_to_nodes.setdefault(support, []).append(idx)
    duplicate_supports = {
        tuple(sorted(support)): tuple(nodes)
        for support, nodes in support_to_nodes.items()
        if len(nodes) > 1
    }

    parents: dict[int, tuple[int, ...]] = {}
    children_lists: dict[int, list[int]] = {idx: [] for idx in range(len(ids))}

    indexed_supports = list(enumerate(supports))
    for child_idx, child_support in indexed_supports:
        candidates = [
            (parent_idx, parent_support)
            for parent_idx, parent_support in indexed_supports
            if child_support < parent_support
        ]
        candidates.sort(key=lambda item: (len(item[1]), item[0]))
        direct_parents: list[int] = []
        direct_parent_supports: list[frozenset[int]] = []
        for parent_idx, parent_support in candidates:
            if any(previous < parent_support for previous in direct_parent_supports):
                continue
            direct_parents.append(parent_idx)
            direct_parent_supports.append(parent_support)
        parents[child_idx] = tuple(direct_parents)
        for parent_idx in direct_parents:
            children_lists[parent_idx].append(child_idx)

    children = {
        idx: tuple(sorted(children_lists[idx], key=lambda child: (len(supports[child]), child)))
        for idx in range(len(ids))
    }

    root_indices = tuple(idx for idx, support in enumerate(supports) if support == full_support)
    leaf_indices = tuple(idx for idx, support in enumerate(supports) if len(support) == 1)
    isolated_indices = tuple(
        idx for idx in range(len(ids)) if not parents[idx] and not children[idx]
    )
    nonroot_without_parent = tuple(
        idx for idx, support in enumerate(supports)
        if support != full_support and not parents[idx]
    )
    nonleaf_without_child = tuple(
        idx for idx, support in enumerate(supports)
        if len(support) > 1 and not children[idx]
    )
    edge_count = int(sum(len(value) for value in parents.values()))
    structure_payload = {
        "ids": ids,
        "bottom_ids": bottom_ids,
        "supports": [sorted(support) for support in supports],
        "parents": {str(idx): list(parent_ids) for idx, parent_ids in parents.items()},
    }
    structure_hash = sha256(
        json.dumps(structure_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return HierarchyGraph(
        ids=ids,
        bottom_ids=bottom_ids,
        supports=supports,
        parents=parents,
        children=children,
        root_indices=root_indices,
        leaf_indices=leaf_indices,
        duplicate_supports=duplicate_supports,
        isolated_indices=isolated_indices,
        nonroot_without_parent=nonroot_without_parent,
        nonleaf_without_child=nonleaf_without_child,
        edge_count=edge_count,
        structure_sha256=structure_hash,
    )


def hierarchy_audit(graph: HierarchyGraph) -> dict:
    support_sizes = np.asarray([len(support) for support in graph.supports], dtype=np.int64)
    return {
        "n_nodes": len(graph.ids),
        "n_bottom": len(graph.bottom_ids),
        "edge_count": graph.edge_count,
        "root_count": len(graph.root_indices),
        "root_indices": list(graph.root_indices),
        "root_ids": [graph.ids[idx] for idx in graph.root_indices],
        "leaf_count": len(graph.leaf_indices),
        "leaf_indices_count_matches_bottom_count": len(graph.leaf_indices) == len(graph.bottom_ids),
        "isolated_count": len(graph.isolated_indices),
        "isolated_indices": list(graph.isolated_indices),
        "nonroot_without_parent_count": len(graph.nonroot_without_parent),
        "nonroot_without_parent_indices": list(graph.nonroot_without_parent),
        "nonleaf_without_child_count": len(graph.nonleaf_without_child),
        "nonleaf_without_child_indices": list(graph.nonleaf_without_child),
        "duplicate_support_count": len(graph.duplicate_supports),
        "duplicate_supports": {
            ",".join(map(str, support)): [graph.ids[idx] for idx in nodes]
            for support, nodes in graph.duplicate_supports.items()
        },
        "support_size_min": int(support_sizes.min(initial=0)),
        "support_size_max": int(support_sizes.max(initial=0)),
        "support_size_mean": float(support_sizes.mean()) if len(support_sizes) else 0.0,
        "parents_by_id": graph.parents_by_id,
        "children_by_id": graph.children_by_id,
        "structure_sha256": graph.structure_sha256,
        "edge_rule": "direct parent = minimal strict support superset derived from S_df only",
    }


def validate_graph_contract(
    graph: HierarchyGraph,
    *,
    fail_on_duplicate_support: bool = False,
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    if not graph.root_indices:
        issues.append({"severity": "FAIL", "code": "HIERARCHY_NO_ROOT"})
    if graph.isolated_indices:
        issues.append({
            "severity": "FAIL",
            "code": "HIERARCHY_ISOLATED_NODE",
            "indices": list(graph.isolated_indices),
        })
    if graph.nonroot_without_parent:
        issues.append({
            "severity": "FAIL",
            "code": "HIERARCHY_NONROOT_WITHOUT_PARENT",
            "indices": list(graph.nonroot_without_parent),
        })
    if graph.nonleaf_without_child:
        issues.append({
            "severity": "FAIL",
            "code": "HIERARCHY_NONLEAF_WITHOUT_CHILD",
            "indices": list(graph.nonleaf_without_child),
        })
    if graph.duplicate_supports:
        issues.append({
            "severity": "FAIL" if fail_on_duplicate_support else "WARN",
            "code": "HIERARCHY_DUPLICATE_SUPPORT",
            "count": len(graph.duplicate_supports),
        })
    return issues


def parents_children_by_id(graph: HierarchyGraph) -> tuple[Mapping[str, list[str]], Mapping[str, list[str]]]:
    return graph.parents_by_id, graph.children_by_id
