import pytest

from sim.radix_tree import RadixTree


def test_lookup_prefix_requires_contiguous_path() -> None:
    tree = RadixTree()
    tree.insert_sequence(["a", "b"], size_bytes=10, current_time=1.0)
    tree.insert_sequence(["x", "c"], size_bytes=10, current_time=2.0)

    depth, nodes = tree.lookup_prefix(["x", "b"])

    assert depth == 1
    assert [node.block_hash for node in nodes] == ["x"]


def test_lookup_does_not_make_plain_insert_sequence_stateful() -> None:
    tree = RadixTree()
    tree.insert_sequence(["a", "b"], size_bytes=10, current_time=1.0)

    depth, _ = tree.lookup_prefix(["a", "b", "c"])
    assert depth == 2

    assert tree.insert_sequence(["c"], size_bytes=10, current_time=2.0) == 1
    assert tree.lookup_prefix(["c"])[0] == 1
    assert tree.lookup_prefix(["a", "b", "c"])[0] == 2


def test_explicit_suffix_insert_extends_matched_path() -> None:
    tree = RadixTree()
    tree.insert_sequence(["a", "b"], size_bytes=10, current_time=1.0)

    full_sequence = ["a", "b", "c", "d"]
    depth, _ = tree.lookup_prefix(full_sequence)
    assert depth == 2

    assert tree.insert_suffix_after_prefix(
        full_sequence,
        depth,
        ["c"],
        size_bytes=10,
        current_time=2.0,
    ) == 1
    assert tree.insert_suffix_after_prefix(
        full_sequence,
        depth + 1,
        ["d"],
        size_bytes=10,
        current_time=3.0,
    ) == 1

    depth, nodes = tree.lookup_prefix(["a", "b", "c", "d"])
    assert depth == 4
    assert [node.block_hash for node in nodes] == ["a", "b", "c", "d"]


def test_suffix_insert_rejects_non_contiguous_hashes() -> None:
    tree = RadixTree()
    tree.insert_sequence(["a", "b"], size_bytes=10, current_time=1.0)

    with pytest.raises(ValueError):
        tree.insert_suffix_after_prefix(
            ["a", "b", "c", "d"],
            2,
            ["d"],
            size_bytes=10,
            current_time=2.0,
        )


def test_ref_count_only_tracks_contiguous_path() -> None:
    tree = RadixTree()
    tree.insert_sequence(["a", "b"], size_bytes=10, current_time=1.0)
    tree.insert_sequence(["x", "b"], size_bytes=10, current_time=2.0)

    tree.acquire_sequence(["x", "b"])

    a = tree.root.children["a"]
    x = tree.root.children["x"]
    assert a.ref_count == 0
    assert a.children["b"].ref_count == 0
    assert x.ref_count == 1
    assert x.children["b"].ref_count == 1

    tree.release_sequence(["x", "b"])
    assert x.ref_count == 0
    assert x.children["b"].ref_count == 0
