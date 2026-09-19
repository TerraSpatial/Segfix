"""Freehand 3D lasso selection for the point cloud.

There is no built-in lasso for a 3D point cloud, so we implement one:

1. While the tool is *armed*, camera rotation/pan/zoom is disabled so a
   left-drag draws a polygon instead of spinning the view.
2. As the user drags, the screen-space path is painted by a transparent Qt
   overlay sitting on top of the canvas.
3. On release we project every 3D point to the same canvas pixel space using
   the live camera transform and test which projections fall inside the drawn
   polygon.  Those points become the new selection.

Because the test is purely in screen space, the lasso grabs every point whose
projection lands inside the outline — including points occluded behind others.
For tree clouds that is usually what you want (sweep a whole stem/canopy
through the foliage); hold Shift to add to an existing selection.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from qtpy.QtWidgets import QWidget


# -- geometry -----------------------------------------------------------
#: Scanline rows are grouped with a 16-bit key so numpy stable-sorts them
#: with a radix pass instead of a comparison sort. Past this many rows the
#: key would not fit and the exact-y sort is used instead -- no canvas is
#: that tall, but a projection can put a stray vertex a long way off it.
_MAX_ROWS = 32767


def _group_by_row(polygon: np.ndarray, ys: np.ndarray):
    """Order ``ys`` by whole scanline row, plus where each row starts.

    Returns ``(order, starts, y0)``, or ``None`` when the outline spans too
    many rows for the 16-bit key.

    Grouping by ``int(y)`` rather than by y itself is what makes this cheap:
    ``np.argsort(..., kind="stable")`` radix-sorts an int16 key and
    comparison-sorts anything wider, which on 2.2M candidate points measured
    43ms against 550ms for the float64 sort the exact ordering needs. An
    edge's band then covers whole rows, and only the two rows at its ends
    hold points that may fall outside it -- so the band test below still
    decides every point on its real y, and the answer is unchanged.
    """
    py = polygon[:, 1]
    y0 = float(np.floor(py.min()))
    rows = int(np.floor(py.max()) - y0) + 1
    if rows > _MAX_ROWS:
        return None
    # Callers have already dropped everything outside the outline's bounding
    # box, so every y lands in [y0, y0 + rows).
    row = (ys - y0).astype(np.int16)
    np.clip(row, 0, rows - 1, out=row)
    order = np.argsort(row, kind="stable")
    starts = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(row, minlength=rows), out=starts[1:])
    return order, starts, y0


def _crossings(polygon: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd ray test over ``pts``, one edge at a time.

    An edge can only be crossed by points whose y lies in that edge's own y
    span, so grouping the points by y once lets each edge do its arithmetic
    on just the contiguous slice it can possibly affect. Summed over a closed
    outline those slices come to roughly two passes over the points however
    many vertices there are, in place of the vertices x points the
    expression-per-edge version does over the whole array.

    That also keeps every temporary the size of one slice rather than the
    size of the cloud, which is what the Windows/Linux split turned on: the
    old version allocated about six full-length arrays per vertex, and a
    large allocate/free round trip measured ~50x dearer on Windows (fresh
    pages through VirtualAlloc, kernel-zeroed) than on Linux (a warm glibc
    arena). Hence 20.0s vs 9.0s there for 3M points and a 500-vertex path.
    """
    n = len(pts)
    inside = np.zeros(n, dtype=bool)
    grouped = _group_by_row(polygon, pts[:, 1])
    if grouped is None:  # pragma: no cover - needs a 32k-row outline
        order = np.argsort(pts[:, 1], kind="stable")
        starts = y0 = None
    else:
        order, starts, y0 = grouped
    xs = np.ascontiguousarray(pts[order, 0])
    ys = np.ascontiguousarray(pts[order, 1])

    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    for ex1, ey1, ex2, ey2 in zip(x1, y1, x2, y2):
        # `(ey1 > y) != (ey2 > y)` is true for exactly min <= y < max, either
        # way the edge runs. A horizontal edge spans nothing, so it crosses
        # no ray -- skipping it here is also what lets the x below divide by
        # the edge's height without guarding it.
        if ey1 == ey2:
            continue
        ylo, yhi = (ey1, ey2) if ey1 < ey2 else (ey2, ey1)
        if starts is None:  # pragma: no cover - the fallback above
            lo = int(np.searchsorted(ys, ylo, side="left"))
            hi = int(np.searchsorted(ys, yhi, side="left"))
        else:
            # Every row the band touches, ends included; the slice is a
            # superset of the band, which `band` below trims to size.
            lo = int(starts[int(ylo - y0)])
            hi = int(starts[int(yhi - y0) + 1])
        if lo >= hi:
            continue
        span_y = ys[lo:hi]
        # x of the edge at each of those heights; the ray crosses when the
        # point sits left of it. Multiply before dividing, so a point sitting
        # exactly on an edge rounds the same way it did when this was one
        # expression -- folding the two constants into a single factor first
        # is a different rounding, and flips such a point in or out.
        x_cross = span_y - ey1
        x_cross *= ex2 - ex1
        x_cross /= ey2 - ey1
        x_cross += ex1
        band = xs[lo:hi] < x_cross
        if starts is not None:
            # Trim the two end rows back to the edge's real span. The exact-y
            # ordering needs no such step, which is the whole of what the
            # cheaper key costs.
            band &= span_y >= ylo
            band &= span_y < yhi
        np.not_equal(inside[lo:hi], band, out=inside[lo:hi])  # flip on crossing

    out = np.empty(n, dtype=bool)
    out[order] = inside
    return out


def points_in_polygon(polygon: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test.

    Parameters
    ----------
    polygon: ``(M, 2)`` array of the closed outline vertices (x, y).
    pts: ``(N, 2)`` array of query points.

    Returns a boolean ``(N,)`` mask of points strictly inside the polygon.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    inside = np.zeros(len(pts), dtype=bool)
    if len(polygon) < 3 or len(pts) == 0:
        return inside

    # A lasso covers a small part of the canvas, so reject on the outline's
    # bounding box first: four cheap comparisons drop most of the cloud, and
    # the O(vertices x points) loop then runs on what is left.
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    px, py = pts[:, 0], pts[:, 1]
    cand = px >= lo[0]
    cand &= px <= hi[0]
    cand &= py >= lo[1]
    cand &= py <= hi[1]
    rows = np.flatnonzero(cand)
    if rows.size:
        inside[rows] = _crossings(polygon, pts[rows])
    return inside


# -- Qt overlay ---------------------------------------------------------
class _LassoOverlay(QWidget):
    """Transparent widget that paints the in-progress lasso path."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._path: list[tuple[float, float]] = []
        self.hide()

    def set_path(self, path) -> None:
        self._path = list(path)
        self.update()

    def clear(self) -> None:
        self._path = []
        self.hide()
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt signature)
        if len(self._path) < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Scale canvas-pixel coords to this widget's logical coords (HiDPI).
        parent = self.parent()
        sw = self.width() / max(parent.width(), 1)
        sh = self.height() / max(parent.height(), 1)
        poly = QPolygonF([QPointF(x * sw, y * sh) for x, y in self._path])

        fill = QPainterPath()
        fill.addPolygon(poly)
        fill.closeSubpath()
        painter.fillPath(fill, QColor(255, 220, 0, 40))

        pen = QPen(QColor(255, 220, 0, 220))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPolyline(poly)
        painter.drawLine(poly.last(), poly.first())  # closing segment


# -- tool ---------------------------------------------------------------
class LassoTool:
    """Manages the armed state, drawing, and selection for the lasso.

    Parameters
    ----------
    view: the :class:`~segfix.cloudview.CloudView`.
    on_select: callback ``(indices: np.ndarray, additive: bool) -> None``
        invoked when a lasso completes, with the selected point indices and
        whether Shift asked to add to the current selection.
    min_points: a drag shorter than this many vertices is treated as a click
        (selection cleared), not a lasso.
    """

    #: Minimum on-screen distance (pixels) between two recorded path
    #: vertices. Every mouse-move event would otherwise append a point
    #: regardless of how far the cursor actually moved; some platforms
    #: (reported: Windows) deliver far more, and far less coalesced,
    #: mouse-move events per drag than others, and with no cap that turns an
    #: ordinary drag into a path with thousands of near-duplicate vertices.
    #: _finish()'s polygon test is O(vertices x loaded points), so that alone
    #: can block the app for seconds on a big loaded cloud -- a few pixels is
    #: well below anything visible in the drawn outline.
    MIN_SEGMENT_PX = 3.0

    def __init__(self, view, on_select, min_points: int = 3):
        self.view = view
        self.on_select = on_select
        self.min_points = min_points
        self._armed = False
        self._overlay = None
        self._path: list[tuple[float, float]] | None = None
        self._additive = False
        self._canvas = view.canvas

    # -- arm / disarm ---------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed

    def _ensure_overlay(self):
        if self._overlay is None:
            # Parent the overlay to the canvas widget itself, so its (0, 0)
            # is the canvas' top-left — the same origin as the vispy mouse
            # ``event.pos`` and the visual→canvas projection. Parenting it to
            # an ancestor instead shifts the drawn outline by the height of
            # whatever docks sit above the canvas.
            self._overlay = _LassoOverlay(self.view.native)
        return self._overlay

    def set_armed(self, armed: bool) -> None:
        if armed == self._armed:
            return
        self._armed = armed
        if armed:
            self.view.set_camera_interactive(False)
            self._connect(True)
            self.view.status = "Lasso armed - drag to select (Shift to add)"
        else:
            self._connect(False)
            self.view.set_camera_interactive(True)
            if self._overlay is not None:
                self._overlay.clear()
            self._path = None
            self.view.status = "Lasso off"

    def toggle(self) -> bool:
        self.set_armed(not self._armed)
        return self._armed

    def reassert(self) -> None:
        """Re-apply the camera lock while armed (kept for call-site parity;
        with vispy the camera doesn't get silently re-enabled, so this is
        just a cheap idempotent nudge)."""
        if self._armed:
            self.view.set_camera_interactive(False)

    def _connect(self, on: bool) -> None:
        events = self._canvas.events
        pairs = (
            (events.mouse_press, self._on_press),
            (events.mouse_move, self._on_move),
            (events.mouse_release, self._on_release),
        )
        for signal, cb in pairs:
            try:
                signal.disconnect(cb)
            except (ValueError, TypeError):
                pass
            if on:
                signal.connect(cb)

    # -- drag handling --------------------------------------------------
    @staticmethod
    def _pos(event) -> tuple[float, float]:
        p = event.pos
        return float(p[0]), float(p[1])

    def _on_press(self, event) -> None:
        if not self._armed or event.button != 1:
            return
        self._additive = "Shift" in getattr(event, "modifiers", ())
        self._path = [self._pos(event)]
        overlay = self._ensure_overlay()
        native = self.view.native
        overlay.setGeometry(0, 0, native.width(), native.height())
        overlay.show()
        event.handled = True

    def _on_move(self, event) -> None:
        if not self._armed or self._path is None:
            return
        pos = self._pos(event)
        last = self._path[-1]
        moved_sq = (pos[0] - last[0]) ** 2 + (pos[1] - last[1]) ** 2
        if moved_sq < self.MIN_SEGMENT_PX ** 2:
            event.handled = True
            return
        self._path.append(pos)
        self._ensure_overlay().set_path(self._path)
        event.handled = True

    def _on_release(self, event) -> None:
        if not self._armed or self._path is None:
            return
        path = np.asarray(self._path, dtype=np.float64)
        self._path = None
        if self._overlay is not None:
            self._overlay.clear()
        self._finish(path, self._additive)
        event.handled = True

    def _finish(self, path: np.ndarray, additive: bool) -> None:
        if len(path) < self.min_points:
            if not additive:
                self.on_select(np.empty(0, dtype=np.int64), additive=False)
            return
        coords = self.view.coords
        # Hidden points (a hide checkbox, "show unassigned", or either section
        # tool — all of which set view.shown) are unselectable, so drop them
        # up front rather than after the test: with a tree isolated out of its
        # neighbours most of the cloud is hidden, and projecting and
        # polygon-testing it only to mask it away afterwards is the bulk of
        # the work for none of the answer.
        shown = np.asarray(self.view.shown, dtype=bool)
        rows = None
        if shown.shape[0] == len(coords) and not shown.all():
            rows = np.flatnonzero(shown)
            coords = coords[rows]
        canvas_xy, valid = self.view.project_to_canvas(coords)
        inside = points_in_polygon(path, canvas_xy)
        inside &= valid
        indices = np.flatnonzero(inside)
        if rows is not None:
            indices = rows[indices]
        self.on_select(indices, additive=additive)


class ClusterTool:
    """Click a point to select the connected patch of same-tree points around
    it; click again in the same spot to loosen the gap and grow it wider.

    A click selects the physically connected blob of the clicked point's own
    tree that the click sits in — so in a closed canopy it isolates one
    continuous lump instead of bleeding into everything it touches.  How big
    a hole counts as "not connected" is the gap setting.  Each further click
    on roughly the same spot, within a couple of seconds, keeps the same seed
    and calls ``on_repeat`` — which the panel wires to "one step looser", the
    same move as the gap slider or the ] key, so the patch grows one notch
    per click and the slider follows.  Move away or pause and the next click
    starts afresh.  Shift adds to the current selection.

    ``grow`` is ``(seed_index) -> np.ndarray`` (supplied by the controller,
    which owns the labels, the gap and the cached KD-tree).
    """

    #: a click within this many pixels of the last one, and within
    #: ``CHAIN_SECONDS`` of it, continues the sequence instead of reseeding
    CHAIN_PIXELS = 30
    CHAIN_SECONDS = 2.5

    def __init__(self, view, grow, on_select, move_tol: int = 6):
        self.view = view
        self.grow = grow
        self.on_select = on_select
        #: fn() called by a repeat click on the same spot; set by the panel.
        #: Nothing happens on a repeat click while it's unset.
        self.on_repeat = None
        self.move_tol = move_tol
        self._armed = False
        self._canvas = view.canvas
        self._press = None
        self._chain = None  # dict(seed, xy, t, base) while a sequence runs

    @property
    def armed(self) -> bool:
        return self._armed

    def set_armed(self, armed: bool) -> None:
        if armed == self._armed:
            return
        self._armed = armed
        self._chain = None
        if armed:
            self.view.set_camera_interactive(False)
            self._connect(True)
            self.view.status = (
                "Cluster - click a point to select its patch, click again "
                "to loosen it (Shift adds)"
            )
        else:
            self._connect(False)
            self.view.set_camera_interactive(True)
            self._press = None
            self.view.status = "Cluster off"

    def toggle(self) -> bool:
        self.set_armed(not self._armed)
        return self._armed

    def _connect(self, on: bool) -> None:
        events = self._canvas.events
        pairs = (
            (events.mouse_press, self._on_press),
            (events.mouse_release, self._on_release),
        )
        for signal, cb in pairs:
            try:
                signal.disconnect(cb)
            except (ValueError, TypeError):
                pass
            if on:
                signal.connect(cb)

    def _on_press(self, event) -> None:
        if not self._armed or event.button != 1:
            return
        p = event.pos
        additive = "Shift" in getattr(event, "modifiers", ())
        self._press = (float(p[0]), float(p[1]), additive)
        event.handled = True

    def _on_release(self, event) -> None:
        if not self._armed or self._press is None:
            return
        px, py, additive = self._press
        self._press = None
        event.handled = True
        dx, dy = float(event.pos[0]) - px, float(event.pos[1]) - py
        if dx * dx + dy * dy > self.move_tol * self.move_tol:
            return  # a drag, not a click — ignore
        self._select_at((px, py), additive)

    def _select_at(self, xy, additive: bool) -> None:
        import time

        now = time.monotonic()
        ch = self._chain
        cont = (
            ch is not None
            and now - ch["t"] <= self.CHAIN_SECONDS
            and (xy[0] - ch["xy"][0]) ** 2 + (xy[1] - ch["xy"][1]) ** 2
            <= self.CHAIN_PIXELS ** 2
        )
        if cont:
            # Same spot again: one step looser, exactly as the slider or ]
            # would do it -- on_repeat moves the gap, and the gap change
            # re-runs this chain through reapply(). The chain keeps its seed
            # and the selection it was added to; only the clock and anchor
            # move, so a run of clicks keeps chaining.
            ch["t"], ch["xy"] = now, (float(xy[0]), float(xy[1]))
            if self.on_repeat is not None:
                self.on_repeat()
            return
        seed = self.view.pick_point(xy)
        if seed is None:
            # Clicked empty space — clear the selection (Shift-click on
            # nothing just does nothing).
            self._chain = None
            if not additive:
                self.on_select(np.empty(0, dtype=np.int64), additive=False)
                self.view.status = "Selection cleared"
            return
        # What the patch is added to: kept on the chain so reapply() can
        # rebuild the selection from scratch -- a patch that has to be able to
        # shrink can't be layered on a selection that already contains it.
        # A copy of the mask, not the live one: reapply() rebuilds the
        # selection from what the click was added to, so it has to survive
        # the selection changing under it.
        base = self.view.selected_mask.copy() if additive else None
        indices = self.grow(seed)
        self._chain = {
            "seed": seed,
            "xy": (float(xy[0]), float(xy[1])),
            "t": now,
            "base": base,
        }
        self.on_select(indices, additive=additive)

    def end_chain(self) -> None:
        """Forget the running click sequence: the next click starts afresh,
        and there's no last click for :meth:`reapply` to re-run."""
        self._chain = None

    def reapply(self) -> bool:
        """Re-run the last click with whatever ``grow`` now returns — called
        when the gap it bridges changes, so the patch on screen tracks the
        setting instead of waiting for another click.

        Rebuilds from the selection the click was added to, not the current
        one, so a tighter gap really does shrink the patch. Returns False
        (and leaves the selection alone) when there's no click to re-run.
        """
        ch = self._chain
        if not self._armed or ch is None:
            return False
        indices = self.grow(ch["seed"])
        base = ch["base"]
        if base is not None and base.any():
            merged = base.copy()
            merged[indices] = True
            indices = merged
        self.on_select(indices, additive=False)
        return True
