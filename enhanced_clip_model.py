import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class CrossAttentionFusion(nn.Module):

    def __init__(self, text_dim, graph_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = text_dim
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, text_dim)
        self.norm = nn.LayerNorm(text_dim)

        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.max_alpha = 0.2

        
    def forward(self, text_features, graph_features):
       
        if not torch.is_tensor(graph_features):
            return text_features

        if torch.isnan(graph_features).any() or torch.isinf(graph_features).any():
            return text_features

        text_proj = self.text_proj(text_features).unsqueeze(1)
        graph_proj = self.graph_proj(graph_features).unsqueeze(1)

        try:
            attended, _ = self.attention(text_proj, graph_proj, graph_proj)
            attended = attended.squeeze(1)
        except Exception:
            return text_features

        residual = self.output_proj(attended)
        residual = torch.tanh(residual)
        gate = torch.clamp(self.alpha, 0.0, self.max_alpha)
        output = text_features + gate * residual
        output = self.norm(output)

        if torch.isnan(output).any() or torch.isinf(output).any():
            return text_features

        return output

class EnhancedCLIP(nn.Module):
    def __init__(self, base_clip_model, graph_encoder, freeze_graph_encoder=False):
        super().__init__()
        self.base_clip = base_clip_model
        self.graph_encoder = graph_encoder
        self.freeze_graph_encoder = freeze_graph_encoder
        
       
        try:
            text_dim = base_clip_model.text.output_dim
        except AttributeError:
            try:
                text_dim = base_clip_model.text_projection.out_features
            except AttributeError:
                text_dim = base_clip_model.text.width

        self.text_graph_fusion = CrossAttentionFusion(
            text_dim=text_dim,
            graph_dim=graph_encoder.out_dim
        )
        
        if freeze_graph_encoder:
            for param in self.graph_encoder.parameters():
                param.requires_grad = False
    
    
    @property
    def logit_scale(self):
        return self.base_clip.logit_scale
    
    def forward(self, images, texts, graphs=None):
        text_features = self.base_clip.encode_text(texts, normalize=True)
        if graphs is not None:
            total_edges = 0
            for et in graphs.canonical_etypes:
                total_edges += graphs.num_edges(et)

            if self.freeze_graph_encoder:
                with torch.no_grad():
                    with autocast(device_type='cuda', enabled=False):
                        graph_features = self.graph_encoder(graphs) if total_edges > 0 else None
            else:
                with autocast(device_type='cuda', enabled=False):
                    graph_features = self.graph_encoder(graphs) if total_edges > 0 else None

            if graph_features is not None:
                enhanced_text_features = self.text_graph_fusion(text_features, graph_features)
            else:
                enhanced_text_features = text_features
        else:
            enhanced_text_features = text_features

        image_features = self.base_clip.encode_image(images, normalize=True)
        
        return {
            "image_features": image_features,
            "text_features": enhanced_text_features,
            "logit_scale": self.base_clip.logit_scale.exp()
        }
    
    def encode_image(self, images, normalize=True):
        return self.base_clip.encode_image(images, normalize)
    
    def encode_text(self, texts, normalize=True):
        return self.base_clip.encode_text(texts, normalize)
    
    def encode_text_with_graph(self, texts,graphs, normalize=True):
        text_features = self.base_clip.encode_text(texts, normalize=True)
        if graphs is not None:
            total_edges = 0
            for et in graphs.canonical_etypes:
                total_edges += graphs.num_edges(et)

            if self.freeze_graph_encoder:
                with torch.no_grad():
                    with autocast(device_type='cuda', enabled=False):
                        graph_features = self.graph_encoder(graphs) if total_edges > 0 else None
            else:
                with autocast(device_type='cuda', enabled=False):
                    graph_features = self.graph_encoder(graphs) if total_edges > 0 else None

            if graph_features is not None:
                enhanced_text_features = self.text_graph_fusion(text_features, graph_features)
            else:
                enhanced_text_features = text_features
        else:
            enhanced_text_features = text_features
        return enhanced_text_features
