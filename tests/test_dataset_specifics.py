import numpy as np
import pytest

from dataloaders.dataset_specifics import get_excluded_slice_indices, get_folds


@pytest.mark.parametrize(
    ('dataset', 'case_count'),
    [('CMR', 35), ('CHAOST2', 20), ('SABS', 30)],
)
def test_query_partitions_cover_each_case_once(dataset, case_count):
    folds = get_folds(dataset)
    query_cases = [case for fold in folds.values() for case in fold[:-1]]

    assert sorted(query_cases) == list(range(case_count))
    assert len(query_cases) == len(set(query_cases))


@pytest.mark.parametrize('dataset', ['CMR', 'CHAOST2', 'SABS'])
def test_each_fold_has_a_dedicated_final_support_case(dataset):
    folds = get_folds(dataset)

    for fold_idx, fold in folds.items():
        next_fold = folds[(fold_idx + 1) % len(folds)]
        assert fold[-1] == next_fold[0]
        assert fold[-1] not in fold[:-1]


def test_setting_two_excludes_slices_containing_any_held_out_class():
    labels = np.array(
        [
            [[0, 0], [0, 0]],
            [[1, 0], [0, 0]],
            [[0, 2], [0, 0]],
            [[1, 2], [0, 0]],
            [[3, 0], [0, 0]],
        ]
    )

    excluded = get_excluded_slice_indices(labels, [1, 2])

    np.testing.assert_array_equal(excluded, [1, 2, 3])


def test_setting_one_does_not_exclude_slices():
    labels = np.ones((2, 2, 2), dtype=np.int64)

    assert get_excluded_slice_indices(labels, None).size == 0
