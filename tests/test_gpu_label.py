"""The "GPU:" status-bar label.

It is the only thing that tells a user whether the discrete GPU was picked
up or whether they are quietly on a software renderer, so "unknown" is a
real answer to get wrong.
"""

from __future__ import annotations

import pytest

from segfix import viewer


# -- when to settle for "unknown" -------------------------------------------
def test_an_answer_is_shown_and_ends_the_asking():
    text, done = viewer.gpu_status("NVIDIA RTX 500 Ada Generation", tries_left=29)
    assert text == "GPU: NVIDIA RTX 500 Ada Generation"
    assert done


def test_no_answer_yet_leaves_the_label_alone_and_asks_again():
    """The bug this exists for: the first draw is not always a context that
    can answer, and giving up on it pinned the label to "unknown" for the
    rest of the session."""
    text, done = viewer.gpu_status(None, tries_left=29)
    assert text is None          # still "detecting..."
    assert not done              # stay hooked to the draw event


def test_it_does_give_up_eventually():
    text, done = viewer.gpu_status(None, tries_left=0)
    assert text == "GPU: unknown"
    assert done


def test_a_late_answer_still_wins():
    """Twenty-nine silent draws then a good one: the label must show it,
    not the "unknown" it was heading for."""
    for remaining in range(29, 0, -1):
        assert viewer.gpu_status(None, remaining) == (None, False)
    assert viewer.gpu_status("Intel Arc", 0) == ("GPU: Intel Arc", True)


def test_an_empty_string_is_not_an_answer():
    # vispy's glGetParameter returns '' rather than None when it has nothing.
    assert viewer.gpu_status("", tries_left=5) == (None, False)
    assert viewer.gpu_status("", tries_left=0) == ("GPU: unknown", True)


# -- where the renderer string comes from ------------------------------------
def test_the_renderer_is_read_through_vispy(monkeypatch):
    """vispy is a hard dependency; PyOpenGL is not a dependency at all, so a
    packaged build has no reason to carry it -- reading through it reported
    "unknown" on every machine the installer shipped to."""
    import sys
    import types

    fake_gl = types.SimpleNamespace(
        GL_RENDERER="GL_RENDERER",
        glGetParameter=lambda name: "Fake Renderer 9000",
    )
    monkeypatch.setitem(sys.modules, "vispy.gloo", types.SimpleNamespace(gl=fake_gl))
    # Make the PyOpenGL fallback explode, so a pass can only come from vispy.
    monkeypatch.setitem(sys.modules, "OpenGL", None)

    assert viewer.gpu_renderer_info() == "Fake Renderer 9000"


def test_no_context_anywhere_reads_as_none(monkeypatch):
    import sys
    import types

    def boom(name):
        raise RuntimeError("no valid context")

    monkeypatch.setitem(
        sys.modules, "vispy.gloo",
        types.SimpleNamespace(gl=types.SimpleNamespace(
            GL_RENDERER="GL_RENDERER", glGetParameter=boom)),
    )
    monkeypatch.setitem(sys.modules, "OpenGL", None)

    assert viewer.gpu_renderer_info() is None
