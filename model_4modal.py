import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSmile4Modal(nn.Module):
    def __init__(self, args, embedding_dims, n_classes_emo):
        super(GraphSmile4Modal, self).__init__()
        self.args = args
        self.text_feature_dim = embedding_dims[0]
        self.audio_feature_dim = embedding_dims[1]
        self.visual_feature_dim = embedding_dims[2]
        self.landmark_feature_dim = embedding_dims[3]
        self.hidden_dim = args.hidden_dim
        self.dropout = args.drop
        self.dim_layer_t = nn.Sequential(nn.Linear(self.text_feature_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout))
        self.dim_layer_a = nn.Sequential(nn.Linear(self.audio_feature_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout))
        self.dim_layer_v = nn.Sequential(nn.Linear(self.visual_feature_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout))
        self.dim_layer_l = nn.Sequential(nn.Linear(self.landmark_feature_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout))
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.emo_output = nn.Linear(self.hidden_dim, n_classes_emo)
        self.landmark_output = nn.Linear(self.hidden_dim, self.landmark_feature_dim)

    def forward(self, feature_t, feature_a, feature_v, feature_l):
        featdim_t = self.dim_layer_t(feature_t)
        featdim_a = self.dim_layer_a(feature_a)
        featdim_v = self.dim_layer_v(feature_v)
        featdim_l = self.dim_layer_l(feature_l)
        fused_features = torch.cat([featdim_t, featdim_a, featdim_v, featdim_l], dim=-1)
        final_feat = self.modal_fusion(fused_features)
        emo_predictions = self.emo_output(final_feat)
        landmark_predictions = self.landmark_output(final_feat)
        return F.log_softmax(emo_predictions, dim=-1), landmark_predictions
