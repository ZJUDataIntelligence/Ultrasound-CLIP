import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLoss(nn.Module):

    
    def __init__(self, args, similarity_weight=1.0, temperature=0.07):

        super().__init__()
        self.similarity_weight = similarity_weight
        self.temperature = temperature
        self.rank = args.rank if hasattr(args, 'rank') else 0
        self.world_size = args.world_size if hasattr(args, 'world_size') else 1
        
    def compute_predicted_similarity_matrix(self, image_features, text_features):

        image_features = torch.nan_to_num(image_features, nan=0.0, posinf=0.0, neginf=0.0)
        text_features = torch.nan_to_num(text_features, nan=0.0, posinf=0.0, neginf=0.0)
        image_features_norm = image_features / (image_features.norm(dim=1, keepdim=True).clamp_min(1e-6))
        text_features_norm  = text_features  / (text_features.norm(dim=1,  keepdim=True).clamp_min(1e-6))

        similarity_matrix = torch.mm(image_features_norm, text_features_norm.t())
        similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return similarity_matrix
    
    def semantic_matching_loss(self, predicted_sim, target_sim):

        target_sim = torch.nan_to_num(target_sim, nan=0.0, posinf=0.0, neginf=0.0)
        predicted_sim = torch.nan_to_num(predicted_sim, nan=0.0, posinf=0.0, neginf=0.0)

        mse_loss = F.mse_loss(predicted_sim, target_sim)

        temp = max(float(self.temperature), 1e-6)
        predicted_logits = (predicted_sim / temp).clamp(min=-50.0, max=50.0)
        target_logits    = (target_sim    / temp).clamp(min=-50.0, max=50.0)
        predicted_prob = F.softmax(predicted_logits, dim=1)
        target_prob    = F.softmax(target_logits, dim=1)
        
        kl_loss = F.kl_div(
            F.log_softmax(predicted_logits, dim=1),
            target_prob,
            reduction='batchmean'
        )
        
        semantic_matching_loss = 0.6 * mse_loss + 0.4 * kl_loss
        
        return semantic_matching_loss
    
    def forward(self, prediction, similarity_matrix):
        if similarity_matrix is None:
            return torch.tensor(0.0, device=prediction['image_features'].device)
        
        image_features = prediction['image_features']
        text_features = prediction['text_features']
        
        predicted_similarity = self.compute_predicted_similarity_matrix(image_features, text_features)
        
        if not (torch.isfinite(predicted_similarity).all() and torch.isfinite(similarity_matrix).all()):
            return torch.zeros([], device=image_features.device, dtype=image_features.dtype)

        semantic_matching_loss = self.semantic_matching_loss(predicted_similarity, similarity_matrix)

        if not torch.isfinite(semantic_matching_loss):
            return torch.zeros([], device=image_features.device, dtype=image_features.dtype)
        
        return semantic_matching_loss
