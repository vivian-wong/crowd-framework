import torch

from st_dif.models.sten import DenseGCNGRU, GCNGRU, KLayerGCNConv


EDGE_INDEX = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)


def test_k1_modes_match():
    torch.manual_seed(3)
    legacy = KLayerGCNConv(1, 2, 4, node_dim=1, gcn_mode="legacy")
    corrected = KLayerGCNConv(1, 2, 4, node_dim=1, gcn_mode="corrected")
    corrected.load_state_dict(legacy.state_dict())
    x = torch.randn(2, 3, 2)
    assert torch.equal(legacy(x, EDGE_INDEX, None), corrected(x, EDGE_INDEX, None))


def test_legacy_runs_one_layer_and_corrected_runs_all_layers():
    x = torch.randn(2, 3, 2)
    legacy = KLayerGCNConv(3, 2, 4, node_dim=1, gcn_mode="legacy")
    corrected = KLayerGCNConv(3, 2, 4, node_dim=1, gcn_mode="corrected")
    corrected.load_state_dict(legacy.state_dict())

    expected = legacy.convs[0](x, EDGE_INDEX, None)
    assert torch.equal(legacy(x, EDGE_INDEX, None), expected)
    assert corrected(x, EDGE_INDEX, None).shape == (2, 3, 4)


def test_corrected_layers_receive_gradients():
    model = KLayerGCNConv(3, 2, 4, node_dim=1, gcn_mode="corrected")
    model(torch.randn(2, 3, 2), EDGE_INDEX, None).sum().backward()
    assert all(conv.lin.weight.grad is not None for conv in model.convs)


def test_paper_parameter_counts_are_unchanged():
    dense = DenseGCNGRU(2, 20, 32, gcn_mode="legacy")
    plain = GCNGRU(2, 20, 32, gcn_mode="legacy")
    assert sum(p.numel() for p in dense.parameters()) == 97_300
    assert sum(p.numel() for p in plain.parameters()) == 96_916


def test_checkpoint_keys_match_between_modes():
    legacy = DenseGCNGRU(2, 3, 2, K=3, D_GCN=4, D_GRU=5,
                         num_gru_layers=1, gcn_mode="legacy")
    corrected = DenseGCNGRU(2, 3, 2, K=3, D_GCN=4, D_GRU=5,
                            num_gru_layers=1, gcn_mode="corrected")
    assert legacy.state_dict().keys() == corrected.state_dict().keys()
    corrected.load_state_dict(legacy.state_dict())


def test_dense_model_batches_match_individual_examples():
    model = DenseGCNGRU(2, 3, 2, K=2, D_GCN=4, D_GRU=5,
                        num_gru_layers=1, gcn_mode="corrected").eval()
    x = torch.randn(2, 3, 2, 3)
    with torch.no_grad():
        batched = model(x, EDGE_INDEX)
        separate = torch.cat([model(x[i:i + 1], EDGE_INDEX) for i in range(2)])
    torch.testing.assert_close(batched, separate)
