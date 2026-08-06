import pytest
import torch

from st_dif.models import DenseGCLSTM, DenseLRGCN, DenseMPNNLSTM, RecurrentGCN
from st_dif.models.baselines import _batch_graph


EDGE_INDEX = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)


def test_batch_graph_repeats_edges_and_attributes():
    weights = torch.arange(4, dtype=torch.float)
    relations = torch.tensor([0, 1, 0, 1])
    edges, batched_weights, batched_relations = _batch_graph(
        EDGE_INDEX, 2, 3, weights, relations,
    )
    expected = torch.cat([EDGE_INDEX, EDGE_INDEX + 3], dim=1)
    assert torch.equal(edges, expected)
    assert torch.equal(batched_weights, weights.repeat(2))
    assert torch.equal(batched_relations, relations.repeat(2))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RecurrentGCN(2, 3, 2, hidden_gcn=4, hidden_rnn=5),
        lambda: DenseGCLSTM(2, 4, 3, K=2),
        lambda: DenseLRGCN(2, 3, 2),
        lambda: DenseMPNNLSTM(2, 3, hidden_size=4),
    ],
)
def test_corrected_batch_matches_separate_graphs(factory):
    model = factory().eval()
    x = torch.randn(2, 3, 2, 3)
    with torch.no_grad():
        batched = model(x, EDGE_INDEX)
        separate = torch.cat([model(x[i:i + 1], EDGE_INDEX) for i in range(2)])
    torch.testing.assert_close(batched, separate, atol=1e-6, rtol=1e-5)


def test_dcrnn_corrected_mode_uses_neighbors():
    torch.manual_seed(7)
    model = RecurrentGCN(2, 3, 1, hidden_gcn=4, hidden_rnn=5).eval()
    x = torch.randn(1, 3, 2, 3)
    alternate = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    with torch.no_grad():
        first = model(x, EDGE_INDEX)
        second = model(x, alternate)
    assert not torch.allclose(first, second)


def test_published_baseline_parameter_counts():
    models = {
        "RecurrentGCN": RecurrentGCN(2, 20, 32, graph_mode="legacy"),
        "DenseGCLSTM": DenseGCLSTM(2, 64, 20, graph_mode="legacy"),
        "DenseLRGCN": DenseLRGCN(2, 20, 32, graph_mode="legacy"),
        "DenseMPNNLSTM": DenseMPNNLSTM(2, 20, graph_mode="legacy"),
    }
    expected = {
        "RecurrentGCN": 122_260,
        "DenseGCLSTM": 51_476,
        "DenseLRGCN": 148,
        "DenseMPNNLSTM": 352_276,
    }
    assert {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    } == expected
