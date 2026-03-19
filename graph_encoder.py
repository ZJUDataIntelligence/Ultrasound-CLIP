import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv, HeteroGraphConv
from .tag_vocab import DIAGNOSIS_VOCAB, DESCRIPTOR_VOCAB
import warnings
warnings.filterwarnings('ignore')

class GraphEncoder(nn.Module):


    def __init__(self, out_dim: int, args=None, hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)

        self.type_embeddings = nn.ParameterDict()

        self.diag_emb = nn.Embedding(len(DIAGNOSIS_VOCAB) + 1, hidden)
        self.desc_emb = nn.Embedding(len(DESCRIPTOR_VOCAB) + 1, hidden)
        self.node_norm = nn.LayerNorm(hidden)

        self.convs = nn.ModuleList([
            HeteroGraphConv({}, aggregate='sum') for _ in range(n_layers)
        ])

        self.proj = nn.Linear(hidden, out_dim)

    def _ensure_type_embeddings(self, g: dgl.DGLHeteroGraph, device):
        for ntype in g.ntypes:
            key = f"emb_{ntype}"
            if key not in self.type_embeddings:
                param = nn.Parameter(torch.zeros(self.hidden, device=device, dtype=torch.float32))
                nn.init.xavier_uniform_(param.unsqueeze(0))
                self.type_embeddings[key] = param

    def _ensure_convs(self, g: dgl.DGLHeteroGraph, device):
        for layer_idx in range(self.n_layers):
            conv: HeteroGraphConv = self.convs[layer_idx]
            if len(conv.mods) == 0:
                mods = {}
                for srctype, etype, dsttype in g.canonical_etypes:
                    mods[etype] = GraphConv(in_feats=self.hidden, out_feats=self.hidden, norm='both', allow_zero_in_degree=True)
                conv.mods = nn.ModuleDict(mods)
                self.convs[layer_idx] = conv.to(device)

    def _infer_batch_size(self, g: dgl.DGLHeteroGraph) -> int:
        if hasattr(g, 'batch_size'):
            return g.batch_size
        try:
            if len(g.ntypes) > 0 and hasattr(g, 'batch_num_nodes'):
                base_ntype = g.ntypes[0]
                return len(g.batch_num_nodes(base_ntype))
        except Exception:
            pass
        return 1

    def _safe_mean_nodes(self, g: dgl.DGLHeteroGraph, ntype: str):
        if ntype not in g.ntypes or g.num_nodes(ntype) == 0:
            return None
        pooled = dgl.mean_nodes(g, 'h', ntype=ntype)
        if pooled is None:
            return None
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0).float()
        return pooled

    def forward(self, g: dgl.DGLHeteroGraph):
        device = next(self.parameters()).device
        g = g.to(device)
        try:
            from torch.amp import autocast
            from contextlib import nullcontext
            autocast_ctx = autocast(device_type='cuda', enabled=False) if device.type == 'cuda' else nullcontext()
        except Exception:
            from contextlib import ExitStack as nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx:
            self._ensure_type_embeddings(g, device)
            self._ensure_convs(g, device)

            try:
                total_edges = 0
                for et in g.canonical_etypes:
                    total_edges += g.num_edges(et)
            except Exception:
                total_edges = 0

            batch_size = self._infer_batch_size(g)
            if total_edges == 0:
                return torch.zeros(batch_size, self.out_dim, device=device, dtype=torch.float32)

            for ntype in g.ntypes:
                num_nodes = g.num_nodes(ntype)
                use_tid = 'tid' in g.nodes[ntype].data and num_nodes > 0
                if use_tid and ntype in ('diagnosis', 'descriptor'):
                    tid = g.nodes[ntype].data['tid'].to(device)
                    if ntype == 'diagnosis':
                        feat = self.node_norm(self.diag_emb(tid).float())
                    else:
                        feat = self.node_norm(self.desc_emb(tid).float())
                    g.nodes[ntype].data['h'] = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    key = f"emb_{ntype}"
                    if key not in self.type_embeddings:
                        param = nn.Parameter(torch.zeros(self.hidden, device=device, dtype=torch.float32))
                        nn.init.xavier_uniform_(param.unsqueeze(0))
                        self.type_embeddings[key] = param
                    type_vec = self.type_embeddings[key]
                    h0 = type_vec.unsqueeze(0).expand(num_nodes, -1).contiguous()
                    g.nodes[ntype].data['h'] = h0.float()

            for layer_idx in range(self.n_layers):
                conv: HeteroGraphConv = self.convs[layer_idx]
                h = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}
                h = conv(g, h)
                for ntype in h:
                    h_nt = torch.relu(h[ntype]).float()
                    h_nt = self.dropout(h_nt)
                    g.nodes[ntype].data['h'] = torch.nan_to_num(h_nt, nan=0.0, posinf=0.0, neginf=0.0)

            pooled_list = []
            for ntype in g.ntypes:
                pooled = self._safe_mean_nodes(g, ntype)
                if pooled is not None:
                    pooled_list.append(pooled)

            if len(pooled_list) == 0:
                return torch.zeros(batch_size, self.out_dim, device=device, dtype=torch.float32)

            graph_h = torch.stack(pooled_list, dim=0).sum(dim=0)
            graph_h = torch.nan_to_num(graph_h, nan=0.0, posinf=0.0, neginf=0.0).float()

            out = self.proj(graph_h)
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).float()
            denom = torch.norm(out, dim=-1, keepdim=True).clamp_min(1e-6)
            out = out / denom
            return out