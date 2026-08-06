"""Graph baselines used in the ST-DIF paper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv
from torch_geometric_temporal.nn.recurrent import DCRNN, GCLSTM


GRAPH_MODES = ("legacy", "corrected")


def _check_mode(mode):
    if mode not in GRAPH_MODES:
        raise ValueError(f"graph_mode must be one of {GRAPH_MODES}; got {mode!r}")


def _batch_graph(edge_index, batch_size, num_nodes, edge_weight=None,
                 edge_type=None):
    """Build a block-diagonal graph for a flattened node batch."""
    edge_count = edge_index.shape[1]
    offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
    batched = edge_index[:, None, :] + offsets[None, :, None]
    batched = batched.reshape(2, batch_size * edge_count)
    if edge_weight is not None:
        edge_weight = edge_weight.repeat(batch_size)
    if edge_type is not None:
        edge_type = edge_type.repeat(batch_size)
    return batched, edge_weight, edge_type


def _graph_inputs(edge_index, batch_size, num_nodes, graph_mode,
                  edge_weight=None, edge_type=None):
    if graph_mode == "legacy":
        return edge_index, edge_weight, edge_type
    return _batch_graph(
        edge_index, batch_size, num_nodes,
        edge_weight=edge_weight, edge_type=edge_type,
    )


class RecurrentGCN(nn.Module):
    """DCRNN followed by the paper's two-layer RNN head."""

    def __init__(self, in_channels, periods, batch_size, hidden_gcn=128,
                 hidden_rnn=64, graph_mode="corrected"):
        super().__init__()
        _check_mode(graph_mode)
        self.periods = periods
        self.batch_size = batch_size
        self.graph_mode = graph_mode
        diffusion_steps = 1 if graph_mode == "legacy" else 2
        self.recurrent = DCRNN(in_channels, hidden_gcn, K=diffusion_steps)
        self.rnn = nn.RNN(
            hidden_gcn, hidden_rnn, num_layers=2,
            nonlinearity="tanh", batch_first=True,
        )
        self.fc = nn.Linear(hidden_rnn, periods)

    def forward(self, X, edge_index, edge_weight=None):
        batch_size, num_nodes, features, steps = X.shape
        graph, weights, _ = _graph_inputs(
            edge_index, batch_size, num_nodes, self.graph_mode, edge_weight,
        )
        sequence = []
        for step in range(steps):
            x = X[:, :, :, step].reshape(batch_size * num_nodes, features)
            sequence.append(self.recurrent(x, graph, weights))
        rnn_input = torch.stack(sequence, dim=1)
        rnn_output, _ = self.rnn(rnn_input)
        return self.fc(rnn_output[:, -1]).view(batch_size, num_nodes, self.periods)


class DenseGCLSTM(nn.Module):
    """GC-LSTM with the common paper input and output shapes."""

    def __init__(self, in_channels, hidden_channels, periods, K=3,
                 normalization="sym", bias=True, graph_mode="corrected"):
        super().__init__()
        _check_mode(graph_mode)
        self.periods = periods
        self.graph_mode = graph_mode
        self.cell = GCLSTM(
            in_channels=in_channels, out_channels=hidden_channels, K=K,
            normalization=normalization, bias=bias,
        )
        self.fc = nn.Linear(hidden_channels, periods)

    def forward(self, X, edge_index, edge_weight=None, lambda_max=None):
        batch_size, num_nodes, features, steps = X.shape
        graph, weights, _ = _graph_inputs(
            edge_index, batch_size, num_nodes, self.graph_mode, edge_weight,
        )
        hidden = cell = None
        for step in range(steps):
            x = X[:, :, :, step].reshape(batch_size * num_nodes, features)
            hidden, cell = self.cell(
                x, graph, edge_weight=weights, H=hidden, C=cell,
                lambda_max=lambda_max,
            )
        return self.fc(hidden).view(batch_size, num_nodes, self.periods)


class DenseLRGCN(nn.Module):
    """Relational GCN-LSTM used for the LRGCN paper baseline."""

    def __init__(self, in_channels, periods, batch_size, num_relations=1,
                 num_bases=1, graph_mode="corrected"):
        super().__init__()
        _check_mode(graph_mode)
        self.in_channels = in_channels
        self.periods = periods
        self.batch_size = batch_size
        self.graph_mode = graph_mode
        args = (in_channels, in_channels, num_relations, num_bases)
        self.conv_x_i, self.conv_h_i = RGCNConv(*args), RGCNConv(*args)
        self.conv_x_f, self.conv_h_f = RGCNConv(*args), RGCNConv(*args)
        self.conv_x_c, self.conv_h_c = RGCNConv(*args), RGCNConv(*args)
        self.conv_x_o, self.conv_h_o = RGCNConv(*args), RGCNConv(*args)
        self.fc = nn.Linear(in_channels, periods)

    def forward(self, X, edge_index, edge_type=None, H=None, C=None):
        batch_size, num_nodes, features, steps = X.shape
        if edge_type is None:
            edge_type = torch.zeros(
                edge_index.shape[1], dtype=torch.long, device=edge_index.device,
            )
        graph, _, relations = _graph_inputs(
            edge_index, batch_size, num_nodes, self.graph_mode,
            edge_type=edge_type,
        )
        state_shape = (batch_size * num_nodes, self.in_channels)
        H = torch.zeros(state_shape, device=X.device) if H is None else H
        C = torch.zeros(state_shape, device=X.device) if C is None else C
        for step in range(steps):
            x = X[:, :, :, step].reshape(batch_size * num_nodes, features)
            input_gate = torch.sigmoid(
                self.conv_x_i(x, graph, relations) + self.conv_h_i(H, graph, relations)
            )
            forget_gate = torch.sigmoid(
                self.conv_x_f(x, graph, relations) + self.conv_h_f(H, graph, relations)
            )
            candidate = torch.tanh(
                self.conv_x_c(x, graph, relations) + self.conv_h_c(H, graph, relations)
            )
            C = forget_gate * C + input_gate * candidate
            output_gate = torch.sigmoid(
                self.conv_x_o(x, graph, relations) + self.conv_h_o(H, graph, relations)
            )
            H = output_gate * torch.tanh(C)
        return self.fc(H).view(batch_size, num_nodes, self.periods)


class DenseMPNNLSTM(nn.Module):
    """Two message-passing layers followed by two LSTM layers."""

    def __init__(self, in_channels, periods, hidden_size=128, dropout=0.0,
                 graph_mode="corrected"):
        super().__init__()
        _check_mode(graph_mode)
        self.hidden_size = hidden_size
        self.periods = periods
        self.dropout = dropout
        self.graph_mode = graph_mode
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.rnn1 = nn.LSTM(2 * hidden_size, hidden_size, 1)
        self.rnn2 = nn.LSTM(hidden_size, hidden_size, 1)
        self.fc = nn.Linear(2 * hidden_size, periods)

    def _graph_layer(self, x, conv, norm, edge_index, edge_weight):
        x = F.relu(conv(x, edge_index, edge_weight))
        x = norm(x)
        return F.dropout(x, p=self.dropout, training=self.training)

    def forward(self, X, edge_index, edge_weight=None):
        batch_size, num_nodes, features, steps = X.shape
        graph, weights, _ = _graph_inputs(
            edge_index, batch_size, num_nodes, self.graph_mode, edge_weight,
        )
        sequence = []
        for step in range(steps):
            x = X[:, :, :, step].reshape(batch_size * num_nodes, features)
            first = self._graph_layer(x, self.conv1, self.bn1, graph, weights)
            second = self._graph_layer(first, self.conv2, self.bn2, graph, weights)
            sequence.append(torch.cat([first, second], dim=1))
        rnn_input = torch.stack(sequence).contiguous()
        rnn_output, (first_hidden, _) = self.rnn1(rnn_input)
        _, (second_hidden, _) = self.rnn2(rnn_output)
        hidden = torch.cat([first_hidden[0], second_hidden[0]], dim=1)
        return self.fc(hidden).view(batch_size, num_nodes, self.periods)
