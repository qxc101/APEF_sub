from __future__ import annotations

import random


def split_indices(
    n: int,
    *,
    val_frac: float = 0.5,
    test_frac: float = 0.0,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    if n < 1:
        return [], [], []
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) > 1.0:
        raise ValueError(f"Invalid split fractions: val={val_frac}, test={test_frac}")

    rng = random.Random(seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)

    test = all_idx[:n_test]
    val = all_idx[n_test : n_test + n_val]
    train = all_idx[n_test + n_val :]
    return train, val, test

