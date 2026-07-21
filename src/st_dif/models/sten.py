import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import *
from torch_geometric_temporal.nn.attention import *
from torch_geometric.nn import GCNConv, CuGraphGATConv
from torch_geometric.nn.models import DeepGCNLayer
'''K stacked GAT layer implementation with pyg https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv
if Multihead attention, we concat outputs (default behavior of pyg implementation).
'''





from torch_geometric_temporal.nn.recurrent import A3TGCN
# NOTE (reproduction fork): the original line was
#   from .evolvegcno import glorot, GCNConv_Fixed_W
# which requires a local copy of evolvegcno.py that is not part of this repo.
# Both symbols are available from torch_geometric_temporal, so we import from there.
from torch_geometric_temporal.nn.recurrent.evolvegcno import glorot, GCNConv_Fixed_W
from torch_geometric.nn import TopKPooling

class A3TGCNWrapper(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        periods: int,
        hidden_channels: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.periods = periods
        self.hidden_channels = hidden_channels

        # A3TGCN already models time internally
        self.recurrent = A3TGCN(
            in_channels,
            hidden_channels,
            periods
        )

        self.fc = torch.nn.Linear(hidden_channels, periods)

    def forward(
        self,
        X: torch.FloatTensor,          # (B, N, F, T)
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        B, N, F, T = X.shape
        assert T == self.periods, "A3TGCN expects T == periods"

        # collect outputs per batch
        out_batch = []

        for b in range(B):
            # A3TGCN expects (N, F, T)
            xb = X[b]                  # (N, F, T)

            h = self.recurrent(
                xb,
                edge_index,
                edge_weight
            )                           # (N, hidden)

            #h = F.relu(h)
            y = self.fc(h)              # (N, periods)
            out_batch.append(y)

        # (B, N, periods)
        return torch.stack(out_batch, dim=0)



class A3TGCNCompatible(torch.nn.Module):
    def __init__(self, in_channels, periods, hidden_channels=128):
        super().__init__()
        self.periods = periods
        self.recurrent = A3TGCN(in_channels, hidden_channels, periods)
        self.fc = torch.nn.Linear(hidden_channels, periods)

    def forward(self, X, edge_index, edge_weight=None):
        B, N, F, T = X.shape
        assert T == self.periods

        # 1) batch edge_index
        E = edge_index.shape[1]
        offsets = (torch.arange(B, device=edge_index.device) * N).view(B, 1, 1)
        edge_index_batched = edge_index.unsqueeze(0).repeat(B, 1, 1) + offsets
        edge_index_batched = edge_index_batched.view(2, B * E)

        # 2) flatten X
        X_batched = X.view(B * N, F, T)

        # 3) run once
        h = self.recurrent(X_batched, edge_index_batched, edge_weight)

        # 4) output
        y = self.fc(h)
        return y.view(B, N, self.periods)


class DenseGCLSTM(torch.nn.Module):
    """
    Temporal wrapper for GCLSTM with DenseGCNGRU-compatible I/O.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        periods: int,
        K: int = 3,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.periods = periods

        self.cell = GCLSTM(
            in_channels=in_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self.fc = torch.nn.Linear(hidden_channels, periods)

    def forward(
        self,
        X: torch.FloatTensor,          # (B, N, F, T)
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:

        B, N, F, T = X.shape
        H, C = None, None

        for t in range(T):
            Xt = X[:, :, :, t].reshape(B * N, F)
            H, C = self.cell(
                Xt,
                edge_index,
                edge_weight=edge_weight,
                H=H,
                C=C,
                lambda_max=lambda_max,
            )

        out = self.fc(H)                         # (B*N, T_out)
        out = out.view(B, N, self.periods, 1)    # (B,N,T_out,1)
        return out.squeeze(-1)                   # (B,N,T_out)


class RecurrentGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        periods: int,
        batch_size: int,
        hidden_gcn: int = 128,
        hidden_rnn: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.periods = periods
        self.batch_size = batch_size

        # spatial modeling
        self.recurrent = DCRNN(
            in_channels,
            hidden_gcn,
            K=1
        )

        # temporal modeling (RNN instead of GRU)
        self.rnn = torch.nn.RNN(
            input_size=hidden_gcn,
            hidden_size=hidden_rnn,
            num_layers=2,
            nonlinearity="tanh",   # or "relu"
            batch_first=True
        )

        self.fc = torch.nn.Linear(hidden_rnn, periods)

    def forward(
        self,
        X: torch.FloatTensor,          # (B, N, F, T)
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        B, N, F, T = X.shape

        gru_in = torch.zeros(
            B, N, T, self.recurrent.out_channels,
            device=X.device
        )

        for t in range(T):
            xt = X[:, :, :, t]          # (B, N, F)
            xt = xt.view(B * N, F)

            gcn_out = self.recurrent(
                xt, edge_index, edge_weight
            )                           # (B*N, hidden_gcn)

            #gcn_out = F.relu(gcn_out)
            gru_in[:, :, t, :] = gcn_out.view(B, N, -1)

        # (B*N, T, hidden_gcn)
        rnn_in = gru_in.flatten(start_dim=0, end_dim=1)

        rnn_out, _ = self.rnn(rnn_in)

        out = self.fc(rnn_out[:, -1, :])  # (B*N, periods)

        out = out.view(B, N, self.periods)

        return out




import torch
from torch_geometric.nn import RGCNConv


class DenseLRGCN(torch.nn.Module):
    r"""Dense LSTM with Relational Graph Convolution (RGCN-LSTM).
    This module is made compatible with DenseGCNGRU style inputs.

    Args:
        in_channels (int): Number of input features.
        periods (int): Number of output time steps.
        batch_size (int): Batch size.
        num_relations (int, optional): Number of edge relation types.
        num_bases (int, optional): Number of bases for RGCN.
    """

    def __init__(
        self,
        in_channels: int,
        periods: int,
        batch_size: int,
        num_relations: int = 1,
        num_bases: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.periods = periods
        self.batch_size = batch_size
        self.num_relations = num_relations
        self.num_bases = num_bases

        # RGCN-LSTM gates
        self.conv_x_i = RGCNConv(in_channels, in_channels, num_relations, num_bases)
        self.conv_h_i = RGCNConv(in_channels, in_channels, num_relations, num_bases)

        self.conv_x_f = RGCNConv(in_channels, in_channels, num_relations, num_bases)
        self.conv_h_f = RGCNConv(in_channels, in_channels, num_relations, num_bases)

        self.conv_x_c = RGCNConv(in_channels, in_channels, num_relations, num_bases)
        self.conv_h_c = RGCNConv(in_channels, in_channels, num_relations, num_bases)

        self.conv_x_o = RGCNConv(in_channels, in_channels, num_relations, num_bases)
        self.conv_h_o = RGCNConv(in_channels, in_channels, num_relations, num_bases)

        self.fc = torch.nn.Linear(in_channels, periods)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.in_channels, device=X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.in_channels, device=X.device)
        return C

    def forward(
        self,
        X: torch.FloatTensor,          # (B, N, F, T)
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor | None = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        B, N, F, T = X.shape

        # Default edge_type = all zeros (single relation)
        if edge_type is None:
            edge_type = torch.zeros(
                edge_index.shape[1], dtype=torch.long, device=edge_index.device
            )

        # Initialize states using first time slice
        Xt0 = X[:, :, :, 0].view(B * N, F)
        H = self._set_hidden_state(Xt0, H)
        C = self._set_cell_state(Xt0, C)

        # Temporal recurrence
        for t in range(T):
            Xt = X[:, :, :, t].view(B * N, F)

            # Input gate
            I = self.conv_x_i(Xt, edge_index, edge_type)
            I = I + self.conv_h_i(H, edge_index, edge_type)
            I = torch.sigmoid(I)

            # Forget gate
            Fg = self.conv_x_f(Xt, edge_index, edge_type)
            Fg = Fg + self.conv_h_f(H, edge_index, edge_type)
            Fg = torch.sigmoid(Fg)

            # Cell update
            Tt = self.conv_x_c(Xt, edge_index, edge_type)
            Tt = Tt + self.conv_h_c(H, edge_index, edge_type)
            Tt = torch.tanh(Tt)

            C = Fg * C + I * Tt

            # Output gate
            O = self.conv_x_o(Xt, edge_index, edge_type)
            O = O + self.conv_h_o(H, edge_index, edge_type)
            O = torch.sigmoid(O)

            H = O * torch.tanh(C)

        # Output
        out = self.fc(H)                # (B*N, periods)
        out = out.view(B, N, self.periods)

        return out



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DenseMPNNLSTM(nn.Module):
    r"""Dense Message Passing Neural Network + LSTM
    Compatible with DenseGCNGRU input format.

    Args:
        in_channels (int): Number of input features.
        hidden_size (int): Dimension of hidden representations.
        periods (int): Number of output time steps.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        periods: int,
        hidden_size: int=128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.periods = periods
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.rnn1 = nn.LSTM(2 * hidden_size, hidden_size, 1)
        self.rnn2 = nn.LSTM(hidden_size, hidden_size, 1)

        self.fc = nn.Linear(2 * hidden_size, periods)

    def _graph_convolution_1(self, X, edge_index, edge_weight):
        X = F.relu(self.conv1(X, edge_index, edge_weight))
        X = self.bn1(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def _graph_convolution_2(self, X, edge_index, edge_weight):
        X = F.relu(self.conv2(X, edge_index, edge_weight))
        X = self.bn2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def forward(
        self,
        X: torch.FloatTensor,          # (B, N, F, T)
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        B, N, F, T = X.shape

        # Prepare tensor for time sequence
        gcn_seq = torch.zeros(B, N, T, self.hidden_size * 2, device=X.device)

        for t in range(T):
            Xt = X[:, :, :, t].reshape(B * N, F)

            X1 = self._graph_convolution_1(Xt, edge_index, edge_weight)
            X2 = self._graph_convolution_2(X1, edge_index, edge_weight)

            Xg = torch.cat([X1, X2], dim=1)          # (B*N, 2*hidden)
            gcn_seq[:, :, t, :] = Xg.view(B, N, -1)

        # LSTM over time
        rnn_in = gcn_seq.flatten(start_dim=0, end_dim=1)  # (B*N, T, 2*hidden)
        rnn_in = rnn_in.transpose(0, 1).contiguous()      # (T, B*N, 2*hidden)

        rnn_out, (h1, _) = self.rnn1(rnn_in)
        rnn_out, (h2, _) = self.rnn2(rnn_out)

        # Hidden states
        H = torch.cat([h1[0], h2[0]], dim=1)  # (B*N, 2*hidden)

        out = self.fc(H)                      # (B*N, periods)
        out = out.view(B, N, self.periods)

        return out



class KLayerGAT(torch.nn.Module): 
    def __init__(self, 
                 K: int, 
                 in_channels: int,  
                 out_channels: int,
                 heads: int, 
                 concat: bool = True,
                 add_self_loops: bool = True): 
        super().__init__()
        self.convs = []
        for k in range(K): 
            if k==0:
                self.convs.append(CuGraphGATConv(in_channels=in_channels,
                                          out_channels=out_channels,
                                          heads=heads,
                                          concat=concat,))
            else: 
                self.convs.append(CuGraphGATConv(in_channels=out_channels,
                                          out_channels=out_channels,
                                          heads=heads,
                                          concat=concat,))
        '''
        Note: saving and loading doesnt work when theres a list like self.convs. 
        So we convert it to nn.sequential. 
        See here: 
            https://discuss.pytorch.org/t/loading-saved-models-gives-inconsistent-results-each-time/36312/24
        '''
        self.convs = torch.nn.Sequential(*self.convs)
    def forward(self, x, edge_index, edge_weight):
        for k in range(len(self.convs)): 
            x = self.convs[k].forward(x, edge_index, edge_weight)
            return x # shape (N, num_heads * out_channels)


''' 
    Helpful module of K stacked GCN layers
'''
class KLayerGCNConv(torch.nn.Module): 
    def __init__(self, 
                 K: int, 
                 in_channels: int,  
                 out_channels: int,
                 node_dim: int,
                 improved: bool = True,
                 cached: bool = False,
                 add_self_loops: bool = True): 
        super().__init__()
        self.convs = []
        for k in range(K): 
            if k==0:
                self.convs.append(GCNConv(in_channels=in_channels,
                                          out_channels=out_channels,
                                          node_dim=node_dim,
                                          improved=improved,
                                          cached=cached,
                                          add_self_loops=add_self_loops))
            else: 
                self.convs.append(GCNConv(in_channels=out_channels,
                                          out_channels=out_channels,
                                          node_dim=node_dim,
                                          improved=improved,
                                          cached=cached,
                                          add_self_loops=add_self_loops))
        '''
        Note: saving and loading doesnt work when theres a list like self.convs. 
        So we convert it to nn.sequential. 
        See here: 
            https://discuss.pytorch.org/t/loading-saved-models-gives-inconsistent-results-each-time/36312/24
        '''
        self.convs = torch.nn.Sequential(*self.convs)
    def forward(self, x, edge_index, edge_weight):
        for k in range(len(self.convs)): 
            x = self.convs[k].forward(x, edge_index, edge_weight)
            return x # shape (N, out_channels)
            


        
'''
    A GCN-GRU model with dense connection, implemented with pyg.DeepGCNLayer 
'''
class DenseGCNGRU(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  
        periods: int, 
        batch_size:int, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super().__init__()

        self.in_channels = in_channels  # 2
        self.periods = periods # 20
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self.densegcn= DeepGCNLayer(conv=KLayerGCNConv(K=3,
                                                       in_channels=self.in_channels,
                                                       out_channels=128,
                                                       node_dim=1,
                                                       improved=self.improved,
                                                       cached=self.cached,
                                                       add_self_loops=self.add_self_loops,
                                                      ),
                                    norm=None,
                                    act=None, #torch.nn.LeakyReLU(),
                                    dropout= 0, #0.1, 
                                    block='dense')
        self.gru = torch.nn.GRU(130,64,2,batch_first=True)
        self.fc = torch.nn.Linear(64, self.periods)
        
    def forward(self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor = None,
               ) -> torch.FloatTensor:
        gru_in = torch.zeros(X.shape[0],X.shape[1],X.shape[3],130).to(X.device) # (B,N,T_in,F_out_GCN)
        for t in range(X.shape[3]):
            gcn_out = self.densegcn(X[:, :, :, t], edge_index, edge_weight) # (B, N, Fout)
            gru_in[:,:,t,:] = gcn_out
        gru_in = gru_in.flatten(start_dim=0, end_dim=1) # (B*N, T_in, F_out_GCN)
        gru_out, _ = self.gru(gru_in) # (B*N,T_in,H)
        out = self.fc(gru_out[:,-1,:]) # (B*N, T_out)
        out = out.view(X.shape[0], X.shape[1], self.periods, -1) # (B,N,T_out,1)
        return out.squeeze(dim=3) # (B,N,T_out)

'''
    Simple GRU model (does not use edge_index)
'''
class GRU_only(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  
        periods: int, 
        batch_size:int, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super().__init__()

        self.in_channels = in_channels  # 2
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self.gru = torch.nn.GRU(self.in_channels,64,2,batch_first=True)
        self.fc = torch.nn.Linear(64, self.periods)

    def forward( self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor = None,  # dummy placeholder
                edge_weight: torch.FloatTensor = None, # dummy placeholder
               ) -> torch.FloatTensor:
        gru_in = torch.reshape(X, (X.shape[0], X.shape[1], self.periods, -1)) #(B,N,2,T)->(B,N,T,2)
        gru_in = gru_in.flatten(start_dim=0, end_dim=1) # (B*N, T, 2)        
        gru_out, _ = self.gru(gru_in) # (B*N,T_in,H)
        out = self.fc(gru_out[:,-1,:]) # (B*N, T_out)
        out = out.view(X.shape[0], X.shape[1], self.periods, -1) # (B,N,T_out,1)
        return out.squeeze(dim=3) # (B,N,T_out)
    
'''
    GCN GRU model without dense connection
'''
class GCNGRU(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  
        periods: int, 
        batch_size:int, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super().__init__()

        self.in_channels = in_channels  # 2
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self.gcns = KLayerGCNConv( K=3,
                                   in_channels=self.in_channels,
                                   out_channels=128,
                                   node_dim=1,
                                   improved=self.improved,
                                   cached=self.cached,
                                   add_self_loops=self.add_self_loops,
                                  )
#         self.gcn1 = GCNConv(
#             in_channels=self.in_channels,
#             out_channels=128,
#             improved=self.improved,
#             cached=self.cached,
#             add_self_loops=self.add_self_loops,
#         )
#         self.gcn2 = GCNConv(
#             in_channels=128,
#             out_channels=128,
#             improved=self.improved,
#             cached=self.cached,
#             add_self_loops=self.add_self_loops,
#         )
#         self.gcn3 = GCNConv(
#             in_channels=128,
#             out_channels=128,
#             improved=self.improved,
#             cached=self.cached,
#             add_self_loops=self.add_self_loops,
#         )
        self.gru = torch.nn.GRU(128,64,2,batch_first=True)
        self.fc = torch.nn.Linear(64, self.periods)

    def forward(self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor = None,
               ) -> torch.FloatTensor:
        gru_in = torch.zeros(X.shape[0],X.shape[1],self.periods,128).to(X.device) # (B,N,T,F_out_GCN)
        for period in range(self.periods):
            gcn_out = self.gcns(X[:,:,:,period], edge_index, edge_weight)                 
#             gcn_out = self.gcn1(X[:, :, :, period], edge_index, edge_weight) # (B, N, Fout)
#             gcn_out = self.gcn2(gcn_out, edge_index, edge_weight) # (B, N, Fout)
#             gcn_out = self.gcn3(gcn_out, edge_index, edge_weight) # (B, N, Fout)
            gru_in[:,:,period,:] = gcn_out
        gru_in = gru_in.flatten(start_dim=0, end_dim=1) # (B*N, T, F_out_GCN)
        gru_out, _ = self.gru(gru_in) # (B*N,T,H)
        out = self.fc(gru_out[:,-1,:]) # (B*N, Tout)
#         out = F.leaky_relu(out)
        out = out.view(X.shape[0], X.shape[1], self.periods, -1) # (B,N,Tout,1)
        return out.squeeze(dim=3) # (B,N,T)




class DenseGCNGRU_FC(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  
        periods: int, 
        batch_size:int, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super().__init__()

        self.in_channels = in_channels  # 2
        self.periods = periods # 20
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self.densegcn= DeepGCNLayer(conv=KLayerGCNConv(K=3,
                                                       in_channels=self.in_channels,
                                                       out_channels=128,
                                                       node_dim=1,
                                                       improved=self.improved,
                                                       cached=self.cached,
                                                       add_self_loops=self.add_self_loops,
                                                      ),
                                    norm=None,
                                    act=None, #torch.nn.LeakyReLU(),
                                    dropout= 0, #0.1, 
                                    block='dense')
        self.gru = torch.nn.GRU(130,64,2,batch_first=True)
        self.proj = torch.nn.Linear(64, 1)
        
    def forward(self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor = None,
               ) -> torch.FloatTensor:
        gru_in = torch.zeros(X.shape[0],X.shape[1],X.shape[3],130).to(X.device) # (B,N,T_in,F_out_GCN)
        for t in range(X.shape[3]):
            gcn_out = self.densegcn(X[:, :, :, t], edge_index, edge_weight) # (B, N, Fout)
            gru_in[:,:,t,:] = gcn_out
        gru_in = gru_in.flatten(start_dim=0, end_dim=1) # (B*N, T_in, F_out_GCN)
        gru_out, _ = self.gru(gru_in) # (B*N,T_in,H)
        out = gru_out.view(X.shape[0], X.shape[1], self.periods, -1) # (B,N,T_out,1)
        out = self.proj(out)
        return out.squeeze(dim=3) # (B,N,T_out)


class DenseGCNGRU_GRU(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  
        periods: int, 
        batch_size:int, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super().__init__()

        self.in_channels = in_channels  # 2
        self.periods = periods # 20
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self.densegcn= DeepGCNLayer(conv=KLayerGCNConv(K=3,
                                                       in_channels=self.in_channels,
                                                       out_channels=1,
                                                       node_dim=1,
                                                       improved=self.improved,
                                                       cached=self.cached,
                                                       add_self_loops=self.add_self_loops,
                                                      ),
                                    norm=None,
                                    act=None, #torch.nn.LeakyReLU(),
                                    dropout= 0, #0.1, 
                                    block='dense')
      #  self.gru = torch.nn.GRU(130,64,2,batch_first=True)
        self.proj = torch.nn.Linear(3, 1)
        
    def forward(self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor = None,
               ) -> torch.FloatTensor:
        gru_in = torch.zeros(X.shape[0],X.shape[1],X.shape[3],3).to(X.device) # (B,N,T_in,F_out_GCN)
        for t in range(X.shape[3]):
            gcn_out = self.densegcn(X[:, :, :, t], edge_index, edge_weight) # (B, N, Fout)
            gru_in[:,:,t,:] = gcn_out
        out = self.proj(gru_in)
        
        return out.squeeze(dim=3) # (B,N,T_out)


#class DenseGCNGRU-GCN(torch.nn.Module):
#    def __init__(
#        self,
#        in_channels: int,  
#        periods: int, 
#        batch_size:int, 
#        improved: bool = False,
#        cached: bool = False,
#        add_self_loops: bool = True):
#        super().__init__()
#
#        self.in_channels = in_channels  # 2
#        self.periods = periods # 20
#        self.improved = improved
#        self.cached = cached
#        self.add_self_loops = add_self_loops
#        self.batch_size = batch_size
#        self._setup_layers()
#
#    def _setup_layers(self):
#        self.densegcn= DeepGCNLayer(conv=KLayerGCNConv(K=3,
#                                                       in_channels=self.in_channels,
#                                                       out_channels=self.period,
#                                                       node_dim=1,
#                                                       improved=self.improved,
#                                                       cached=self.cached,
#                                                       add_self_loops=self.add_self_loops,
#                                                      ),
#                                    norm=None,
#                                    act=None, #torch.nn.LeakyReLU(),
#                                    dropout= 0, #0.1, 
#                                    block='dense')
#        self.gru = torch.nn.GRU(130,64,2,batch_first=True)
#        self.fc = torch.nn.Linear(64, self.periods)
#        
#    def forward(self, 
#                X: torch.FloatTensor,
#                edge_index: torch.LongTensor, 
#                edge_weight: torch.FloatTensor = None,
#               ) -> torch.FloatTensor:
#        out = torch.zeros(X.shape[0],X.shape[1],X.shape[3],130).to(X.device) # (B,N,T_in,F_out_GCN)
#        return out.squeeze(dim=3) # (B,N,T_out)



###################### Dongdong Wang 01-13-2026
#
#
#
#
#import torch
#from torch import nn
#from torch_geometric.nn import TransformerConv
#
#class S_DenseGCNTransformer(nn.Module):
#    def __init__(self, node_features=2, hidden_dim=32, out_features=2, heads=4, seq_len=6):
#        super().__init__()
#        self.seq_len = seq_len
#
#        # Spatial graph conv per timestep
#        self.spatial_conv = TransformerConv(node_features, hidden_dim, heads=heads)
#
#        # Temporal transformer: models node features across time
#        self.temporal_transformer = nn.Transformer(
#            d_model=hidden_dim*heads,
#            nhead=4,
#            num_encoder_layers=2,
#            dim_feedforward=128,
#            batch_first=True
#        )
#
#        # Output layer
#        self.fc = nn.Linear(hidden_dim*heads, out_features)
#
#        self.spatial_conv = TransformerConv(
#            in_channels=20,   # <- must match your dataset's node features
#            out_channels=128, # hidden size
#            heads=4,          # optional
#            )
#
#
#    def forward(self, x, edge_index):
#        """
#        x: [batch, seq_len, num_nodes, node_features]
#        edge_index: [2, num_edges]
#        """
#        batch_size, seq_len, num_nodes, feat = x.shape
#        spatial_out = []
#
#        # Apply graph conv to each timestep
#        for t in range(seq_len):
#            # flatten batch & nodes for TransformerConv
#            h = self.spatial_conv(x[:, t, :, :].reshape(-1, feat), edge_index)
#            # reshape back to [batch, num_nodes, hidden]
#            spatial_out.append(h.view(batch_size, num_nodes, -1))
#
#        # Stack across time: [batch, num_nodes, seq_len, hidden]
#        spatial_out = torch.stack(spatial_out, dim=2)  # [B, N, T, H]
#
#        # Flatten nodes into batch dimension for transformer
#        spatial_out = spatial_out.reshape(batch_size*num_nodes, seq_len, -1)  # [B*N, T, H]
#
#        # Temporal transformer
#        temporal_out = self.temporal_transformer(spatial_out)  # [B*N, T, H]
#
#        # Take last time step
#        temporal_out = temporal_out[:, -1, :]  # [B*N, H]
#
#        # Fully connected to output
#        out = self.fc(temporal_out)  # [B*N, out_features]
#
#        # Reshape back to [batch, num_nodes, out_features]
#        out = out.view(batch_size, num_nodes, -1)
#        return out
#
#
#
#
#class DenseGCNTransformer(nn.Module):
#    def __init__(self, node_features, hidden_dim, out_features, heads=2):
#        super().__init__()
#        # Spatial graph conv per timestep
#        self.spatial_conv = TransformerConv(node_features, hidden_dim, heads=heads)
#
#        # Temporal transformer encoder: models node features across time
#        encoder_layer = nn.TransformerEncoderLayer(
#            d_model=hidden_dim*heads,
#            nhead=4,
#            dim_feedforward=128,
#            dropout=0.1,
#            batch_first=True
#        )
#        self.temporal_transformer = nn.TransformerEncoder(
#            encoder_layer,
#            num_layers=2
#        )
#
#        # Output layer
#        self.fc = nn.Linear(hidden_dim*heads, out_features)
#
#    def forward(self, x, edge_index):
#        """
#        x: [batch, seq_len, num_nodes, node_features]
#        edge_index: [2, num_edges]
#        """
#        batch_size, seq_len, num_nodes, feat = x.shape
#        spatial_out = []
#
#        # Apply graph conv to each timestep
#        for t in range(seq_len):
#            h = self.spatial_conv(x[:, t, :, :].reshape(-1, feat), edge_index)
#            spatial_out.append(h.view(batch_size, num_nodes, -1))
#
#        # Stack over time: [batch, num_nodes, seq_len, hidden]
#        spatial_out = torch.stack(spatial_out, dim=2)  # [B, N, T, H]
#
#        # Flatten batch and nodes for TransformerEncoder
#        B, N, T, H = spatial_out.shape
#        spatial_out = spatial_out.view(B*N, T, H)  # [B*N, seq_len, hidden]
#
#        # Temporal modeling
#        temporal_out = self.temporal_transformer(spatial_out)  # [B*N, seq_len, hidden]
#
#        # Take the last time step prediction
#        out = self.fc(temporal_out[:, -1, :])  # [B*N, out_features]
#
#        # Reshape back to [B, N, out_features]
#        out = out.view(B, N, -1)
#
#        return out

