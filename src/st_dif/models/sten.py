import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, CuGraphGATConv
from torch_geometric.nn.models import DeepGCNLayer
'''K stacked GAT layer implementation with pyg https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv
if Multihead attention, we concat outputs (default behavior of pyg implementation).
'''
GCN_MODES = ("legacy", "corrected")


def _validate_gcn_mode(mode):
    if mode not in GCN_MODES:
        raise ValueError(f"gcn_mode must be one of {GCN_MODES}; got {mode!r}")


class KLayerGAT(torch.nn.Module): 
    def __init__(self, 
                 K: int, 
                 in_channels: int,  
                 out_channels: int,
                 heads: int, 
                 concat: bool = True,
                 add_self_loops: bool = True,
                 gcn_mode: str = "corrected"):
        super().__init__()
        if K < 1:
            raise ValueError("K must be at least 1")
        _validate_gcn_mode(gcn_mode)
        self.gcn_mode = gcn_mode
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
        layers_to_run = 1 if self.gcn_mode == "legacy" else len(self.convs)
        for k in range(layers_to_run):
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
                 add_self_loops: bool = True,
                 gcn_mode: str = "corrected"):
        super().__init__()
        if K < 1:
            raise ValueError("K must be at least 1")
        _validate_gcn_mode(gcn_mode)
        self.gcn_mode = gcn_mode
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
        layers_to_run = 1 if self.gcn_mode == "legacy" else len(self.convs)
        for k in range(layers_to_run):
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
        add_self_loops: bool = True,
        K: int = 3,
        D_GCN: int = 128,
        D_GRU: int = 64,
        num_gru_layers: int = 2,
        gcn_mode: str = "corrected"):
        super().__init__()

        self.in_channels = in_channels  # 2
        self.periods = periods # 20
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self.K = K
        self.D_GCN = D_GCN
        self.D_GRU = D_GRU
        self.num_gru_layers = num_gru_layers
        self.gcn_mode = gcn_mode
        self._setup_layers()

    def _setup_layers(self):
        self.densegcn= DeepGCNLayer(conv=KLayerGCNConv(K=self.K,
                                                       in_channels=self.in_channels,
                                                       out_channels=self.D_GCN,
                                                       node_dim=1,
                                                       improved=self.improved,
                                                       cached=self.cached,
                                                       add_self_loops=self.add_self_loops,
                                                       gcn_mode=self.gcn_mode,
                                                      ),
                                    norm=None,
                                    act=None, #torch.nn.LeakyReLU(),
                                    dropout= 0, #0.1, 
                                    block='dense')
        self.gru = torch.nn.GRU(
            self.D_GCN + self.in_channels,
            self.D_GRU,
            self.num_gru_layers,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(self.D_GRU, self.periods)
        
    def forward(self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor = None,
               ) -> torch.FloatTensor:
        gru_in = torch.zeros(
            X.shape[0], X.shape[1], X.shape[3],
            self.D_GCN + self.in_channels,
            device=X.device,
        )
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
        add_self_loops: bool = True,
        K: int = 3,
        D_GCN: int = 128,
        D_GRU: int = 64,
        num_gru_layers: int = 2,
        gcn_mode: str = "corrected"):
        super().__init__()

        self.in_channels = in_channels  # 2
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self.K = K
        self.D_GCN = D_GCN
        self.D_GRU = D_GRU
        self.num_gru_layers = num_gru_layers
        self.gcn_mode = gcn_mode
        self._setup_layers()

    def _setup_layers(self):
        self.gcns = KLayerGCNConv( K=self.K,
                                   in_channels=self.in_channels,
                                   out_channels=self.D_GCN,
                                   node_dim=1,
                                   improved=self.improved,
                                   cached=self.cached,
                                   add_self_loops=self.add_self_loops,
                                   gcn_mode=self.gcn_mode,
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
        self.gru = torch.nn.GRU(
            self.D_GCN,
            self.D_GRU,
            self.num_gru_layers,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(self.D_GRU, self.periods)

    def forward(self, 
                X: torch.FloatTensor,
                edge_index: torch.LongTensor, 
                edge_weight: torch.FloatTensor = None,
               ) -> torch.FloatTensor:
        gru_in = torch.zeros(
            X.shape[0], X.shape[1], self.periods, self.D_GCN, device=X.device
        )
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
