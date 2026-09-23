# segfix — fix instance segmentation of tree point clouds.
# Copyright (C) 2026 Tim Devereux, The University of Queensland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json

import numpy as np
import pytest

from segfix import workspace


def test_create_workspace_copies_file_and_writes_manifest(tmp_path):
    source = tmp_path / "source" / "cloud.ply"
    source.parent.mkdir()
    source.write_bytes(b"fake ply data")

    ws_dir = tmp_path / "MyProject"
    data_path = workspace.create_workspace(source, ws_dir)

    assert data_path == ws_dir / "cloud.ply"
    assert data_path.read_bytes() == b"fake ply data"
    assert workspace.data_file(ws_dir) == data_path
    manifest = json.loads((ws_dir / workspace.MANIFEST_NAME).read_text())
    assert manifest["source"] == str(source.resolve())


def test_create_workspace_is_independent_copy(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"original")
    ws_dir = tmp_path / "proj"

    data_path = workspace.create_workspace(source, ws_dir)
    data_path.write_bytes(b"edited")

    # The copy changed; the source did not.
    assert source.read_bytes() == b"original"
    assert data_path.read_bytes() == b"edited"


def test_create_workspace_refuses_nonempty_existing_dir(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    ws_dir = tmp_path / "proj"
    ws_dir.mkdir()
    (ws_dir / "something.txt").write_text("already here")

    with pytest.raises(FileExistsError):
        workspace.create_workspace(source, ws_dir)


# -- progress ---------------------------------------------------------------
def test_copy_reports_progress_from_nothing_to_done(tmp_path):
    """The bar has to start below 1 and finish at 1: a single report of 1.0
    at the end is the frozen window this exists to replace."""
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"x" * (workspace._COPY_CHUNK * 3 + 17))
    seen = []

    workspace.create_workspace(
        source, tmp_path / "proj", report=lambda m, f: seen.append((m, f))
    )

    assert [m for m, _ in seen] == ["Copying"] * 4
    fractions = [f for _, f in seen]
    assert fractions[0] < 1.0
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)


def test_copy_without_a_report_still_copies(tmp_path):
    """Headless callers and the tests pass nothing."""
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"payload" * 1000)
    data = workspace.create_workspace(source, tmp_path / "proj")
    assert data.read_bytes() == source.read_bytes()


def test_empty_source_reports_completion_rather_than_dividing_by_zero(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"")
    seen = []
    data = workspace.create_workspace(
        source, tmp_path / "proj", report=lambda m, f: seen.append((m, f))
    )
    assert data.read_bytes() == b""
    assert seen == []  # nothing to read, so nothing to report


def _laz_source(path, n=5000):
    """An arbor-shaped .laz: Extra-Bytes treeID, RGB, and an EVLR."""
    import laspy
    from laspy.vlrs.vlrlist import VLRList

    header = laspy.LasHeader(version="1.4", point_format=7)
    header.offsets = [400000.0, 6000000.0, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="treeID", type=np.int32, description="Unique ID per tree",
    ))
    rng = np.random.default_rng(0)
    coords = rng.random((n, 3)) * [50.0, 50.0, 30.0] + [400000.0, 6000000.0, 0.0]
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    las.treeID = rng.integers(0, 40, n).astype(np.int64)
    las.red = np.arange(n, dtype=np.uint16)
    las.evlrs = VLRList([laspy.VLR(
        user_id="segfix", record_id=1, description="note", record_data=b"hi",
    )])
    las.write(str(path))
    return las


def test_laz_import_decompresses_to_an_uncompressed_las(tmp_path):
    import laspy

    source = tmp_path / "plot.laz"
    _laz_source(source)

    data = workspace.create_workspace(source, tmp_path / "proj")

    assert data.name == "plot.las"
    # Uncompressed and memory-mappable is the whole point of the conversion.
    with laspy.open(str(data)) as reader:
        assert not reader.header.are_points_compressed


def test_laz_import_keeps_every_field_header_and_evlr(tmp_path):
    """Streaming the points through in chunks must lose nothing that
    laspy.read(src).write(dest) would have carried."""
    import laspy

    source = tmp_path / "plot.laz"
    _laz_source(source)
    original = laspy.read(str(source))

    data = workspace.create_workspace(source, tmp_path / "proj")
    copied = laspy.read(str(data))

    assert copied.header.point_count == original.header.point_count
    assert str(copied.header.point_format) == str(original.header.point_format)
    assert list(copied.header.scales) == list(original.header.scales)
    assert list(copied.header.offsets) == list(original.header.offsets)
    assert list(copied.header.mins) == list(original.header.mins)
    assert list(copied.header.maxs) == list(original.header.maxs)
    assert len(copied.evlrs) == len(original.evlrs) == 1
    assert copied.evlrs[0].record_data == b"hi"
    for field in ("X", "Y", "Z", "treeID", "red"):
        assert np.array_equal(getattr(copied, field), getattr(original, field))


def test_laz_import_reports_progress_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_LAZ_CHUNK", 1000)
    source = tmp_path / "plot.laz"
    _laz_source(source, n=5000)
    seen = []

    workspace.create_workspace(
        source, tmp_path / "proj", report=lambda m, f: seen.append((m, f))
    )

    assert [m for m, _ in seen] == ["Decompressing"] * 5
    fractions = [f for _, f in seen]
    assert fractions[0] == pytest.approx(0.2)
    assert fractions[-1] == pytest.approx(1.0)
    assert fractions == sorted(fractions)


# -- remembered decisions ----------------------------------------------------
def test_settings_are_empty_until_something_is_decided(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    data = workspace.create_workspace(source, tmp_path / "proj")
    assert workspace.settings(data) == {}


def test_a_declined_answer_is_remembered_as_declined(tmp_path):
    """The point of the whole exercise: "no, don't downsample" has to be
    distinguishable from "never asked", or reopening asks again."""
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    data = workspace.create_workspace(source, tmp_path / "proj")

    workspace.remember(data, voxel_size=None)

    assert workspace.settings(data) == {"voxel_size": None}
    assert "voxel_size" in workspace.settings(data)


def test_remember_merges_rather_than_replacing(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    data = workspace.create_workspace(source, tmp_path / "proj")

    workspace.remember(data, spacing=0.005)
    workspace.remember(data, voxel_size=0.02)

    assert workspace.settings(data) == {"spacing": 0.005, "voxel_size": 0.02}
    # and the rest of the manifest is untouched
    manifest = json.loads((tmp_path / "proj" / workspace.MANIFEST_NAME).read_text())
    assert manifest["source"] == str(source.resolve())
    assert manifest["data_file"] == "cloud.ply"


def test_a_cloud_outside_a_project_has_nowhere_to_remember(tmp_path):
    """A file opened straight off disk has no manifest; every one of these
    has to be a no-op rather than an error."""
    loose = tmp_path / "loose.ply"
    loose.write_bytes(b"data")
    assert workspace.settings(loose) == {}
    workspace.remember(loose, voxel_size=0.02)       # must not raise
    assert workspace.settings(loose) == {}
    assert workspace.cached_array(loose, "voxel") is None
    workspace.cache_array(loose, "voxel", np.arange(3))  # must not raise


def test_unreadable_settings_fall_back_to_asking(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    data = workspace.create_workspace(source, tmp_path / "proj")
    (tmp_path / "proj" / workspace.MANIFEST_NAME).write_text("{ not json")
    assert workspace.settings(data) == {}


# -- derived caches ----------------------------------------------------------
def test_a_cached_array_comes_back_unchanged(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    data = workspace.create_workspace(source, tmp_path / "proj")
    value = np.array([3, 1, 4, 1, 5], dtype=np.int64)

    workspace.cache_array(data, "voxel", value)

    np.testing.assert_array_equal(workspace.cached_array(data, "voxel"), value)
    assert (tmp_path / "proj" / workspace.CACHE_DIR).is_dir()


def test_a_cache_is_dropped_when_the_data_file_changes(tmp_path):
    """Keyed on the bytes it was derived from, so a file replaced underneath
    recomputes instead of handing back indices into the old one."""
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    data = workspace.create_workspace(source, tmp_path / "proj")
    workspace.cache_array(data, "voxel", np.arange(5))
    assert workspace.cached_array(data, "voxel") is not None

    data.write_bytes(b"different content entirely")

    assert workspace.cached_array(data, "voxel") is None


def test_caches_of_different_kinds_do_not_collide(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    data = workspace.create_workspace(source, tmp_path / "proj")
    workspace.cache_array(data, "voxel", np.arange(4))
    workspace.cache_array(data, "other", np.arange(9))
    assert len(workspace.cached_array(data, "voxel")) == 4
    assert len(workspace.cached_array(data, "other")) == 9


def test_a_corrupt_cache_reads_as_absent(tmp_path):
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")
    data = workspace.create_workspace(source, tmp_path / "proj")
    workspace.cache_array(data, "voxel", np.arange(5))
    cache = tmp_path / "proj" / workspace.CACHE_DIR / "cloud.ply.voxel.npz"
    cache.write_bytes(b"not an npz")
    assert workspace.cached_array(data, "voxel") is None
