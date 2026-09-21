"""Export every tree in the cloud as its own file.

One file per tree, in the format the project's cloud is in — a tree out of a
LAS is a LAS, out of a PLY a PLY — carrying every field the source has and
its own coordinates, so an exported tree still lines up with the original
cloud in any other tool (a session's global shift is a view convenience and
is never written; see :mod:`treecatalog`).

Trees are read at full resolution *from the file*, not from the working set:
a downsampled session edits one point per voxel, but an exported tree should
be the whole tree. Unsaved edits are therefore not in the export, which is
why the menu offers to save first.
"""

from __future__ import annotations

import os

import numpy as np

from .model import NOISE, UNASSIGNED
from .treecatalog import ProgressFn


def tree_path(out_dir: str, source_path: str, label: int) -> str:
    """Where tree ``label`` is written: the cloud's own name and extension
    with the tree's ID appended, e.g. ``plot_a.las`` -> ``plot_a_tree_7.las``.
    """
    stem, ext = os.path.splitext(os.path.basename(source_path))
    return os.path.join(out_dir, f"{stem}_tree_{label}{ext}")


def group_rows(labels: np.ndarray) -> dict[int, np.ndarray]:
    """``{label: file rows}`` for the real trees in ``labels``, lowest ID
    first. Unassigned points and anything dismissed as noise are not trees
    and are left out.

    One sort for the whole cloud rather than a scan per tree: a plot with a
    thousand trees would otherwise walk every point a thousand times.
    """
    order = np.argsort(labels, kind="stable")
    uniq, starts, counts = np.unique(
        labels[order], return_index=True, return_counts=True
    )
    return {
        int(label): order[start:start + count]
        for label, start, count in zip(uniq.tolist(), starts.tolist(),
                                       counts.tolist())
        if label not in (UNASSIGNED, NOISE)
    }


def export_trees(
    catalog, out_dir: str, progress: ProgressFn | None = None,
    only: set[int] | None = None,
) -> list[str]:
    """Write every tree in ``catalog``'s file to ``out_dir``, one file each —
    or, given ``only``, just the trees whose IDs are in it (a Done list, say;
    IDs no longer in the file, because a tree was merged away, are ignored).

    Returns the paths written, in tree-ID order. ``progress``
    (:data:`~segfix.treecatalog.ProgressFn`) is called per tree, which is
    what makes a big plot's export legible.
    """
    os.makedirs(out_dir, exist_ok=True)
    if progress is not None:
        progress("Reading tree labels…", 0.0)
    groups = group_rows(catalog.file_labels())
    if only is not None:
        groups = {label: rows for label, rows in groups.items() if label in only}
    written: list[str] = []
    if not groups:
        return written
    with catalog.subset_writer() as write:
        for done, (label, rows) in enumerate(groups.items()):
            if progress is not None:
                progress(
                    f"Tree {label} ({rows.size:,} points)…", done / len(groups)
                )
            dest = tree_path(out_dir, catalog.path, label)
            write(np.sort(rows), dest)  # file order, so the copy reads linearly
            written.append(dest)
    if progress is not None:
        progress("Finishing…", 1.0)
    return written
