import torch

from torch_scatter import scatter_add, scatter


def test_scatter_add_groups_values_by_index() -> None:
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    index = torch.tensor([0, 1, 0, 2])
    assert torch.equal(scatter_add(src, index, dim_size=3), torch.tensor([4.0, 2.0, 4.0]))


def test_scatter_mean_reduces_groups() -> None:
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    index = torch.tensor([0, 1, 0, 2])
    assert torch.equal(scatter(src, index, reduce="mean", dim_size=3), torch.tensor([2.0, 2.0, 4.0]))
