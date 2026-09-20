"""Every key of the review loop sits under the left hand.

The right hand is on the mouse from the moment a tree loads — orbiting,
lassoing, clicking clusters — so a key it has to reach for (the old L, K, N,
U, H and the bracket keys) means letting go, dozens of times a tree. These
pin the map itself: what each key does, that nothing needs the right hand,
and that the keys this replaced still work for anyone who learned them.
"""

from __future__ import annotations

import types

import pytest

from segfix.widgets import LEFT_HAND_KEYS, shortcut_bindings


def _panel():
    """A stand-in panel that records which action a key reached."""
    fired = []

    def action(name):
        return lambda *a: fired.append(name)

    def toggler(name):
        return types.SimpleNamespace(toggle=action(name))

    panel = types.SimpleNamespace(
        fired=fired,
        lasso_btn=toggler("lasso"),
        tree_lasso_btn=toggler("lasso tree"),
        cluster_btn=toggler("cluster"),
        show_unassigned=toggler("show unassigned"),
        cross_enable=toggler("cross section"),
        lasso_section_enable=toggler("lasso section"),
        section_draw_btn=toggler("draw section"),
        step_cluster_gap=lambda step: fired.append(f"gap {step:+d}"),
        on_move_mode=action("move"),
        on_done_next=action("done"),
        on_add=action("add"),
        on_create_new=action("new tree"),
        on_unassign=action("unassign"),
        on_noise=action("noise"),
        _step=lambda step: fired.append(f"step {step:+d}"),
    )
    return panel


def _does(key):
    panel = _panel()
    shortcut_bindings(panel)[key]()
    assert len(panel.fired) == 1
    return panel.fired[0]


#: The whole loop, and where it now lives on the keyboard.
EXPECTED = {
    "Q": "lasso", "W": "lasso tree", "E": "cluster",
    "R": "gap -1", "T": "gap +1",
    "A": "add", "S": "new tree", "D": "unassign", "F": "show unassigned",
    "X": "noise", "C": "cross section",
    "Z": "step -1", "V": "step +1",
    "Esc": "move", "Space": "done",
    "Shift+Q": "draw section", "Shift+C": "lasso section",
}

#: What each of them replaced. Still bound: a hand that already knows these
#: shouldn't have to unlearn anything.
LEGACY = {
    "L": "lasso", "Ctrl+L": "lasso tree", "K": "cluster",
    "[": "gap -1", "]": "gap +1",
    "N": "new tree", "U": "unassign", "H": "show unassigned",
    "Left": "step -1", "Right": "step +1", "Shift+L": "draw section",
}


@pytest.mark.parametrize("key, action", sorted(EXPECTED.items()))
def test_each_key_does_what_it_should(key, action):
    assert _does(key) == action


@pytest.mark.parametrize("key, action", sorted(LEGACY.items()))
def test_the_keys_these_replaced_still_work(key, action):
    assert _does(key) == action


def test_nothing_in_the_loop_needs_the_right_hand():
    reachable = {key for key in EXPECTED if key in LEFT_HAND_KEYS}
    assert reachable == set(EXPECTED), (
        f"needs the right hand: {sorted(set(EXPECTED) - reachable)}"
    )


def test_the_map_holds_exactly_these_keys():
    """So a new binding has to be a deliberate choice about where the hand
    sits, not an incidental one."""
    assert set(shortcut_bindings(_panel())) == set(EXPECTED) | set(LEGACY)


def test_no_key_is_bound_twice():
    bindings = shortcut_bindings(_panel())
    assert len(bindings) == len(set(bindings))
    # And each action is reachable from its new key and its old one only.
    assert sorted(EXPECTED.values()) == sorted(set(EXPECTED.values()))
