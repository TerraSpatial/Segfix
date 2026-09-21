"""Exporting every tree as its own file.

The promise: one file per real tree, in the source format, at full
resolution, with every field and the file's own coordinates — whatever the
session was doing (downsampled, globally shifted) and whatever is unsaved.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from segfix import export, io
from segfix.model import NOISE, UNASSIGNED, PointCloud
from segfix.treecatalog import open_catalog


def _ply(tmp_path, coords, labels, name="plot.ply"):
    cloud = PointCloud(coords=coords.astype(np.float32),
                       labels=np.asarray(labels, np.int32))
    cloud.attributes["intensity"] = np.arange(len(coords), dtype=np.uint16)
    path = str(tmp_path / name)
    io.save(cloud, path)
    return path


def _las(tmp_path, coords, labels, name="plot.las", offsets=(0.0, 0.0, 0.0)):
    import laspy

    header = laspy.LasHeader(version="1.4", point_format=7)
    header.offsets = list(offsets)
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    las.treeID = np.asarray(labels, dtype=np.int64)
    las.intensity = np.arange(len(coords), dtype=np.uint16)
    path = str(tmp_path / name)
    las.write(path)
    return path


def _grid(n_per_tree, labels, step=1.0):
    """``(coords, labels)`` for ``labels``, each tree its own block."""
    coords, out = [], []
    for k, lab in enumerate(labels):
        base = np.arange(n_per_tree) * step
        coords.append(np.column_stack([base + k * 100, base, base]))
        out.append(np.full(n_per_tree, lab))
    return np.concatenate(coords).astype(np.float64), np.concatenate(out)


# -- naming and grouping ------------------------------------------------------
def test_each_tree_is_named_after_the_cloud_and_keeps_its_extension():
    # Joined with the platform's own separator, so the expectation is too --
    # spelling it "/out/..." passed on Linux and failed on Windows, where
    # the maintainer actually runs the suite.
    assert export.tree_path("/out", "/proj/plot_a.las", 7) == os.path.join(
        "/out", "plot_a_tree_7.las"
    )
    assert export.tree_path("/out", "/proj/plot_a.ply", 12) == os.path.join(
        "/out", "plot_a_tree_12.ply"
    )


def test_unassigned_and_noise_are_not_trees():
    labels = np.array([UNASSIGNED, 2, NOISE, 1, 2, UNASSIGNED], np.int32)
    groups = export.group_rows(labels)
    assert list(groups) == [1, 2]
    np.testing.assert_array_equal(np.sort(groups[2]), [1, 4])


# -- PLY ----------------------------------------------------------------------
def test_ply_export_writes_one_file_per_tree_with_every_field(tmp_path):
    coords, labels = _grid(5, [1, 2, 5])
    labels[:2] = UNASSIGNED  # ground: not a tree, not exported
    path = _ply(tmp_path, coords, labels)
    cat = open_catalog(path)

    written = export.export_trees(cat, str(tmp_path / "trees"))

    assert [os.path.basename(p) for p in written] == [
        "plot_tree_1.ply", "plot_tree_2.ply", "plot_tree_5.ply",
    ]
    whole = io.load(path)
    for label, dest in zip((1, 2, 5), written):
        tree = io.load(dest)
        rows = np.flatnonzero(whole.labels == label)
        assert tree.n_points == rows.size
        np.testing.assert_array_equal(tree.labels, whole.labels[rows])
        np.testing.assert_allclose(tree.coords, whole.coords[rows])
        np.testing.assert_array_equal(
            tree.attributes["intensity"], whole.attributes["intensity"][rows]
        )


def test_ply_export_fixes_the_vertex_count_in_the_header(tmp_path):
    coords, labels = _grid(4, [1, 2])
    cat = open_catalog(_ply(tmp_path, coords, labels))
    [tree1, _] = export.export_trees(cat, str(tmp_path / "trees"))
    with open(tree1, "rb") as fh:
        header = fh.read(400)
    assert b"element vertex 4\n" in header


def test_a_ply_with_faces_is_refused(tmp_path):
    coords, labels = _grid(3, [1])
    path = _ply(tmp_path, coords, labels)
    with open(path, "rb") as fh:
        raw = fh.read()
    raw = raw.replace(b"end_header\n", b"element face 2\nproperty list uchar int vertex_indices\nend_header\n")
    faced = tmp_path / "mesh.ply"
    faced.write_bytes(raw)
    cat = open_catalog(str(faced))
    with pytest.raises(ValueError, match="'face' data"):
        export.export_trees(cat, str(tmp_path / "trees"))


# -- LAS ----------------------------------------------------------------------
def test_las_export_keeps_coordinates_extra_bytes_and_extents(tmp_path):
    import laspy

    coords, labels = _grid(6, [1, 3])
    coords += np.array([500000.0, 7000000.0, 0.0])  # UTM-sized, as in the field
    path = _las(tmp_path, coords, labels, offsets=(500000.0, 7000000.0, 0.0))
    cat = open_catalog(path)

    written = export.export_trees(cat, str(tmp_path / "trees"))

    assert len(written) == 2
    for label, dest in zip((1, 3), written):
        tree = laspy.read(dest)
        rows = np.flatnonzero(labels == label)
        assert tree.header.point_count == rows.size
        np.testing.assert_array_equal(np.asarray(tree.treeID), labels[rows])
        np.testing.assert_allclose(np.asarray(tree.x), coords[rows, 0], atol=1e-3)
        np.testing.assert_allclose(np.asarray(tree.y), coords[rows, 1], atol=1e-3)
        np.testing.assert_array_equal(
            np.asarray(tree.intensity), np.arange(len(coords))[rows]
        )
        # Extents describe this tree, not the plot it came out of.
        np.testing.assert_allclose(tree.header.mins, coords[rows].min(axis=0),
                                   atol=1e-3)
        np.testing.assert_allclose(tree.header.maxs, coords[rows].max(axis=0),
                                   atol=1e-3)


def test_a_globally_shifted_session_still_exports_real_coordinates(tmp_path):
    coords, labels = _grid(5, [1, 2])
    coords += np.array([500000.0, 7000000.0, 0.0])
    path = _las(tmp_path, coords, labels, offsets=(500000.0, 7000000.0, 0.0))
    cat = open_catalog(path, shift_prompt=lambda mins, maxs, suggested: suggested)
    assert cat.global_shift is not None  # the session works near the origin

    import laspy

    [tree1, _] = export.export_trees(cat, str(tmp_path / "trees"))
    rows = np.flatnonzero(labels == 1)
    np.testing.assert_allclose(np.asarray(laspy.read(tree1).x), coords[rows, 0],
                               atol=1e-3)


# -- what the file holds, at full resolution ----------------------------------
def test_a_downsampled_session_exports_whole_trees(tmp_path):
    coords, labels = _grid(400, [1, 2], step=0.005)  # 5 mm: dense enough to offer
    path = _ply(tmp_path, coords, labels)
    cat = open_catalog(
        path, density_prompt=lambda spacing, n_points, suggested, kept: suggested
    )
    assert cat.is_decimated and cat.working_count < cat.count

    written = export.export_trees(cat, str(tmp_path / "trees"))

    for label, dest in zip((1, 2), written):
        assert io.load(dest).n_points == 400  # every point, not one per voxel


def test_unsaved_edits_are_not_exported_but_saved_ones_are(tmp_path):
    coords, labels = _grid(5, [1, 2])
    cat = open_catalog(_ply(tmp_path, coords, labels))
    assert not cat.has_unsaved_edits()

    cat.labels[cat.indices_for(2)] = 1  # merge 2 into 1, in the session only
    assert cat.has_unsaved_edits()
    [before] = export.export_trees(cat, str(tmp_path / "before"))[:1]
    assert io.load(before).n_points == 5

    cat.save()
    assert not cat.has_unsaved_edits()
    written = export.export_trees(cat, str(tmp_path / "after"))
    assert len(written) == 1 and io.load(written[0]).n_points == 10


def test_progress_is_reported_per_tree(tmp_path):
    coords, labels = _grid(3, [1, 2, 3])
    cat = open_catalog(_ply(tmp_path, coords, labels))
    seen = []
    export.export_trees(cat, str(tmp_path / "trees"),
                        progress=lambda msg, frac: seen.append((msg, frac)))
    assert [msg for msg, _ in seen][1:-1] == [
        "Tree 1 (3 points)…", "Tree 2 (3 points)…", "Tree 3 (3 points)…",
    ]
    assert [round(f, 2) for _, f in seen] == [0.0, 0.0, 0.33, 0.67, 1.0]


# -- RGB-segmented PLY (raycloudtools) ---------------------------------------
def test_rgb_segmented_ply_exports_by_colour(tmp_path):
    """No label column: the tree is the point's colour, so an exported tree
    is the points of one colour, written back in the same encoding."""
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    colours = [(10, 20, 30), (40, 50, 60), (0, 0, 0)]  # black = unsegmented
    arr = np.zeros(9, dtype=dtype)
    for k in range(9):
        arr["x"][k], arr["y"][k], arr["z"][k] = k, k * 2, k * 3
        arr["red"][k], arr["green"][k], arr["blue"][k] = colours[k % 3]
    path = tmp_path / "rays.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(arr)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + arr.tobytes())

    cat = open_catalog(str(path))
    written = export.export_trees(cat, str(tmp_path / "trees"))

    assert len(written) == 2  # the black points are unsegmented, not a tree
    for dest, colour in zip(written, colours[:2]):
        raw = Path(dest).read_bytes()
        start = raw.index(b"end_header\n") + len(b"end_header\n")
        rows = np.frombuffer(raw[start:], dtype=dtype)
        assert rows.size == 3
        assert {tuple(int(v) for v in (r["red"], r["green"], r["blue"]))
                for r in rows} == {colour}


# -- the menu action ----------------------------------------------------------
def test_the_menu_handler_saves_first_then_writes_every_tree(tmp_path, monkeypatch):
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QFileDialog, QMessageBox

    from segfix import app

    coords, labels = _grid(4, [1, 2])
    cat = open_catalog(_ply(tmp_path, coords, labels))
    cat.labels[cat.indices_for(2)] = 1  # an unsaved merge

    class _View:
        status = ""

    view = _View()

    class _Panel:
        c = type("C", (), {"view": view})()
        saved = False

        def on_save(self):
            type(self).saved = True
            cat.save()

    out = tmp_path / "trees"
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Save)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(out))
    # The handler imports the runner itself, so patch it at source: this
    # test is about save-then-export, not about the worker thread, and
    # standing in for the runner keeps it free of a QApplication.
    from segfix import progress_ui

    monkeypatch.setattr(
        progress_ui, "run_with_progress",
        lambda parent, title, detail, work: work(
            lambda *a: None, lambda fn, *args: fn(*args)
        ),
    )

    app._export_trees(None, _Panel(), cat)

    assert _Panel.saved  # the edit was saved before reading the file
    written = sorted(p.name for p in out.iterdir())
    assert written == ["plot_tree_1.ply"]  # tree 2 was merged into 1
    assert io.load(str(out / "plot_tree_1.ply")).n_points == 8
    assert "Exported 1 tree(s)" in view.status


# -- just the trees marked Done (issue #3) --------------------------------------
def test_export_can_take_just_a_subset(tmp_path):
    coords, labels = _grid(3, [1, 2, 5])
    cat = open_catalog(_ply(tmp_path, coords, labels))
    written = export.export_trees(cat, str(tmp_path / "out"), only={2, 5, 99})
    # 99 isn't in the file any more (merged away, say): quietly skipped.
    assert [Path(p).name for p in written] == ["plot_tree_2.ply", "plot_tree_5.ply"]


def test_the_done_list_is_read_from_the_progress_sidecar(tmp_path):
    import json

    from segfix.scene_ui import read_done

    cloud = tmp_path / "plot.ply"
    assert read_done(str(cloud)) == set()                    # no sidecar yet
    Path(f"{cloud}.segfix.json").write_text(json.dumps({"done": [3, 1]}))
    assert read_done(str(cloud)) == {1, 3}
    Path(f"{cloud}.segfix.json").write_text("{not json")
    assert read_done(str(cloud)) == set()                    # unreadable: none


@pytest.fixture
def qapp():
    """A QApplication that outlives the test body: one with no surviving
    reference is collected at once, and the next widget crashes."""
    os = __import__("os")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _run_export_menu(tmp_path, monkeypatch, done, choice):
    """Drive File > Export Trees... with ``done`` marked, answering the
    which-trees question with the button whose text starts ``choice``.
    Returns (files written, whether the question was asked)."""
    import json

    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QFileDialog, QMessageBox

    from segfix import app, progress_ui

    coords, labels = _grid(3, [1, 2, 5])
    path = _ply(tmp_path, coords, labels)
    if done:
        Path(f"{path}.segfix.json").write_text(json.dumps({"done": sorted(done)}))
    cat = open_catalog(path)

    asked = []

    def answer(box):
        asked.append(box.text())
        for button in box.buttons():
            if button.text().replace("&", "").startswith(choice):
                button.click()
                return 0
        raise AssertionError(f"no {choice!r} button in {[b.text() for b in box.buttons()]}")

    out = tmp_path / "trees"
    monkeypatch.setattr(QMessageBox, "exec", answer)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
    monkeypatch.setattr(
        progress_ui, "run_with_progress",
        lambda parent, title, detail, work: work(
            lambda *a: None, lambda fn, *args: fn(*args)
        ),
    )
    panel = type("P", (), {"c": type("C", (), {"view": type("V", (), {"status": ""})()})(),
                           "on_save": lambda self: None})()
    app._export_trees(None, panel, cat)
    written = sorted(p.name for p in out.iterdir()) if out.exists() else []
    return written, asked


def test_only_the_trees_marked_done(tmp_path, monkeypatch, qapp):
    written, asked = _run_export_menu(tmp_path, monkeypatch, {1, 5}, "Only the 2")
    assert written == ["plot_tree_1.ply", "plot_tree_5.ply"]
    assert asked == ["2 of 3 trees are marked Done. Which do you want to export?"]


def test_or_all_of_them(tmp_path, monkeypatch, qapp):
    written, _asked = _run_export_menu(tmp_path, monkeypatch, {1, 5}, "All 3")
    assert written == ["plot_tree_1.ply", "plot_tree_2.ply", "plot_tree_5.ply"]


def test_cancel_exports_nothing(tmp_path, monkeypatch, qapp):
    written, _asked = _run_export_menu(tmp_path, monkeypatch, {1}, "Cancel")
    assert written == []


def test_with_nothing_done_there_is_no_question(tmp_path, monkeypatch, qapp):
    written, asked = _run_export_menu(tmp_path, monkeypatch, set(), "unused")
    assert asked == []
    assert len(written) == 3
