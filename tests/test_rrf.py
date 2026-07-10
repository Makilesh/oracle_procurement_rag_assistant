from core.retrieval import rrf_fuse


def test_rrf_agreement_wins() -> None:
    # "b" is ranked well by both lists; "a" and "c" only by one each.
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_dedup() -> None:
    fused = rrf_fuse([["a", "b"], ["a", "b"]], k=60)
    assert fused == ["a", "b"]
    assert len(fused) == len(set(fused))


def test_rrf_rank_order_within_single_list() -> None:
    fused = rrf_fuse([["x", "y", "z"]], k=60)
    assert fused == ["x", "y", "z"]


def test_rrf_top_of_one_list_beats_tail_of_both() -> None:
    # "top" is #1 in list one and absent from list two; "mid" is low in both.
    fused = rrf_fuse([["top", "f1", "f2", "f3", "mid"], ["f4", "f5", "f6", "f7", "mid"]], k=60)
    assert fused.index("top") < fused.index("f2")


def test_rrf_empty_lists() -> None:
    assert rrf_fuse([[], []], k=60) == []
