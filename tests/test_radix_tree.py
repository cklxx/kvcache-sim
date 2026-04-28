from sim.radix_tree import RadixTree


def test_lookup_prefix_requires_contiguous_path() -> None:
    tree = RadixTree()
    tree.insert_sequence(["a", "b"], size_bytes=10, current_time=1.0)
    tree.insert_sequence(["x", "c"], size_bytes=10, current_time=2.0)

    depth, nodes = tree.lookup_prefix(["x", "b"])

    assert depth == 1
    assert [node.block_hash for node in nodes] == ["x"]


def test_lookup_then_suffix_insert_extends_matched_path() -> None:
    tree = RadixTree()
    tree.insert_sequence(["a", "b"], size_bytes=10, current_time=1.0)

    depth, _ = tree.lookup_prefix(["a", "b", "c", "d"])
    assert depth == 2

    assert tree.insert_sequence(["d"], size_bytes=10, current_time=2.0) == 0
    assert tree.insert_sequence(["c"], size_bytes=10, current_time=3.0) == 2

    depth, nodes = tree.lookup_prefix(["a", "b", "c", "d"])
    assert depth == 4
    assert [node.block_hash for node in nodes] == ["a", "b", "c", "d"]


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
