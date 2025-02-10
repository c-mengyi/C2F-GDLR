import torch
import torch.nn as nn
import torch.nn.functional as F


class softmax_head(nn.Module):
    def __init__(self, feat_dim, num_cls):
        super(softmax_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))

    def forward(self, x, label):
        logit = torch.mm(x, self.weight)
        return logit, logit


class normface_head(nn.Module):
    def __init__(self, feat_dim, num_cls, s=32):
        super(normface_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        x_norm = F.normalize(x,dim=1)
        w_norm = F.normalize(self.weight,dim=0)
        cosine = torch.mm(x_norm, w_norm).clamp(-1, 1)
        logit = self.s * cosine
        return logit, cosine


# CosFace head
class cosface_head(nn.Module):
    def __init__(self, feat_dim, num_cls, s=32, m=0.35):
        super(cosface_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=0)
        cosine = torch.mm(x_norm, w_norm).clamp(-1, 1)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        logit = self.s * (cosine - one_hot * self.m)

        return logit, cosine


class cdpl_head(nn.Module):
    def __init__(self, feat_dim, num_cls, s=32, m=0.35, a=2,t=0,r=2):
        super(cdpl_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.m = m
        self.a = a
        self.t=t
        self.r=r
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        x_norm = F.normalize(x, dim=1)  # Normalize input features
        w_norm = F.normalize(self.weight, dim=0)  # Normalize weights
        cosine = torch.mm(x_norm, w_norm).clamp(-1, 1)  # Compute cosine similarities

        # One-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        one_hot_inverse = 1.0 - one_hot

        # Calculate cosine_label: the cosine values corresponding to the true labels
        cosine_label = cosine.gather(1, label.view(-1, 1)).view(-1)

        # Calculate cos_sec: the maximum cosine value per row, excluding cosine_label
        # Mask the true label positions to -inf to avoid considering them
        masked_cosine = cosine.masked_fill(one_hot.bool(), float('-inf'))
        cos_sec, _ = masked_cosine.max(dim=1)

        # Calculate cos_sub: cosine_label minus cos_sec
        cos_sub = cosine_label - cos_sec

        # Calculate r: exp(a * cos_sub)
        beta = torch.exp(self.r * cos_sub)

        cosine_label_expanded = cosine_label.unsqueeze(1).expand_as(cosine)

        cos_sub_m = cosine_label_expanded - cosine
        cosine_margin = torch.zeros_like(cosine, dtype=torch.float32)
        cosine_margin[cos_sub_m > self.t] = (cosine[cos_sub_m > self.t] + 1).pow(self.a)

        # Update the logits: logit = s * (cosine - one_hot * m) * r
        # Apply the multiplier only to the true class logits
        logit = self.s * (cosine - one_hot * self.m) + one_hot_inverse * cosine_margin
        beta = beta.to(logit.dtype)

        # Apply the multiplier only to the true class logits without inplace operation
        updated_logit = logit.scatter(1, label.view(-1, 1), logit.gather(1, label.view(-1, 1)) * beta.view(-1, 1))


        return updated_logit, cosine
