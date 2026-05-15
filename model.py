import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


class AttentionPooling(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.score = nn.Linear(dim_in, 1)

    def forward(self, x, valid_mask=None):
        # x: (B, N, D), valid_mask: (B, N) with True for valid tokens
        attn_logits = self.score(self.norm(x)).squeeze(-1)  # (B, N)

        if valid_mask is None:
            attn_weight = torch.softmax(attn_logits, dim=1)
        else:
            attn_logits = attn_logits.masked_fill(~valid_mask, -1e9)
            attn_weight = torch.softmax(attn_logits, dim=1)
            attn_weight = attn_weight * valid_mask.to(attn_weight.dtype)
            attn_weight = attn_weight / attn_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return torch.sum(x * attn_weight.unsqueeze(-1), dim=1)


class StrokeEmbeddingSequence(nn.Module):
    # KAGEストロークの特徴量の埋め込みを作成
    def __init__(self, num_stroke_classes, num_startpoint_classes, num_endpoint_classes, dim_model, effective_resolution):
        super().__init__()

        # embeddings
        dim_embed = dim_model  # label embeddingの次元数
        self.stroke_embed = nn.Embedding(num_stroke_classes, dim_embed)
        self.startpoint_embed = nn.Embedding(num_startpoint_classes, dim_embed)
        self.endpoint_embed = nn.Embedding(num_endpoint_classes, dim_embed)

        # positional encodings
        # 系列上の位置ではなく、座標上の位置をエンコードする役目であることに注意
        dim_pe = effective_resolution  # positional encodingの次元数
        self.pe_coeffs = torch.linspace(start=1.0, end=dim_pe, steps=dim_pe)[None, None, None, :]  # (1, 1, 1, Dpe)  = [[[[1, 2, 3, ...., Dpe]]]]となる
        # 次元を揃えるためのLayer
        self.pe_mixer = nn.Linear(16*dim_pe, dim_model)  # ストロークの代表点の数4 x 2次元の座標をエンコードするため、16*dim_peを追加

    def forward(self, labels, control_points):
        # labels: (B, Nstrokes, 3)
        # control_points: (B, Nstrokes, 4, 2)
        
        # mask for padding
        src_key_padding_mask = labels[:, :, 0] < 0  # (B, Nstrokes) True部分をマスク

        # process labels
        # maximum()はパディング判別用のマイナス値を取り除く役割
        x_stroke_emb = self.stroke_embed(labels[:, :, 0].clamp_min(0))  # (B, Nstrokes) -> (B, Nstrokes, Dembed)
        x_startpoint_emb = self.startpoint_embed(labels[:, :, 1].clamp_min(0))  # (B, Nstrokes) -> (B, Nstrokes, Dembed)
        x_endpoint_emb = self.endpoint_embed(labels[:, :, 2].clamp_min(0))  # (B, Nstrokes) -> (B, Nstrokes, Dembed)

        # 制御点の位置エンコード
        # 一般的なpositional encodingと係数が異なるので注意
        self.pe_coeffs = self.pe_coeffs.to(control_points.device)  # (1, 1, 1, Dpe)
        control_points_flat = control_points.view(control_points.size(0), control_points.size(1), control_points.size(2)*control_points.size(3), 1)  # (B, Nstrokes, 8, 1)
        pe_sin = torch.sin(torch.pi*self.pe_coeffs*control_points_flat)  # (B, Nstrokes, 8, Dpe)
        pe_cos = torch.cos(torch.pi*self.pe_coeffs*control_points_flat)  # (B, Nstrokes, 8, Dpe)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)  # (B, Nstrokes, 8, Dpe*2)
        pe_flat = pe.view(pe.size(0), pe.size(1), -1)  # (B, Nstrokes, 8*Dpe*2)
        x_pe = self.pe_mixer(pe_flat)  # (B, Nstrokes, Dmodel)

        # 統合
        x = (x_stroke_emb + x_startpoint_emb + x_endpoint_emb + x_pe)/4**0.5  # (B, Nstrokes, Dmodel)

        return x, src_key_padding_mask

class StrokeSetEncoder(nn.Module):
    # ストローク集合の特徴量の埋め込みを作成
    def __init__(self, dim_emb, tf_dim_model, tf_layers, tf_heads, tf_dropout, tf_dim_ff):
        super().__init__()

        # stroke encoder
        num_stroke_classes = 6
        num_startpoint_classes = 7
        num_endpoint_classes = 12
        effective_resolution = 32
        self.stroke_embedding = StrokeEmbeddingSequence(num_stroke_classes, num_startpoint_classes, num_endpoint_classes, dim_model=tf_dim_model, effective_resolution=effective_resolution)
        self.stroke_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=tf_dim_model, nhead=tf_heads, dim_feedforward=tf_dim_ff, dropout=tf_dropout, norm_first=True, activation="gelu", batch_first=True), num_layers=tf_layers)

        # 集約用のMLP
        self.output_mlp = nn.Linear(tf_dim_model, dim_emb)
        self.attention_pool = AttentionPooling(dim_emb)
    
    def forward(self, labels, control_points):
        x_stroke_embedding, src_key_padding_mask = self.stroke_embedding(labels, control_points)  # (B, Nstrokes, Dmodel)
        x_stroke_feature = self.stroke_transformer(x_stroke_embedding, src_key_padding_mask=src_key_padding_mask)  # (B, Nstrokes, Dmodel)

        x_stroke_feature = self.output_mlp(x_stroke_feature)  # (B, Nstrokes, Dmodel)
        
        valid_mask = ~src_key_padding_mask  # (B, Nstrokes) - True=有効なデータ
        x_stroke_feature = self.attention_pool(x_stroke_feature, valid_mask=valid_mask)  # (B, Dmodel)

        return x_stroke_feature

class ImageEncoder(nn.Module):
    def __init__(self, dim_emb, vit_layers):
        super().__init__()

        # dinov2
        self.vit_layers = vit_layers
        vit_mean = (0.485, 0.456, 0.406)
        vit_std = (0.229, 0.224, 0.225)
        self.normalize = transforms.Normalize(mean=vit_mean, std=vit_std)
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.vit.requires_grad_(False)
        self.vit.eval()

        # 集約用MLP/Attention Pooling
        vit_dim = self.vit.embed_dim
        self.output_mlp = nn.Linear(vit_dim, dim_emb)
        self.attention_pool = AttentionPooling(dim_emb)
        
    def forward(self, x):
        x = torch.concatenate([x]*3, dim=1)
        xn = self.normalize(x)

        assert(self.vit.chunked_blocks == False)
        x = self.vit.prepare_tokens_with_masks(xn)
        for i in range(self.vit_layers):
            x = self.vit.blocks[i](x)
        # remove REG/CLS tokens
        x = x[:, 1:, :]
        x = self.output_mlp(x)
        x = self.attention_pool(x)
        return x

class DualEncoder(nn.Module):
    def __init__(self, image_encoder, text_encoder, proj_dim=256):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def forward(self, image, text):
        hi = self.image_encoder(image)
        ht = self.text_encoder(text)
        return hi, ht


if __name__ == "__main__":
    import torchinfo

    # 適当なエンコーダー
    dim_emb = 256

    tf_dim_model = 256
    tf_dim_ff = 256
    tf_layers = 4
    tf_heads = 8
    tf_dropout = 0.1
    stroke_set_encoder = StrokeSetEncoder(dim_emb, tf_dim_model, tf_layers, tf_heads, tf_dropout, tf_dim_ff)

    vit_layers = 4
    image_encoder = ImageEncoder(dim_emb, vit_layers)

    # モデルのサマリー
    batch_size = 4
    num_strokes = 10
    stroke_labels = torch.randint(0, 5, (batch_size, num_strokes, 3))  # (B, Nstrokes, 3)
    control_points = torch.rand(batch_size, num_strokes, 4, 2)  # (B, Nstrokes, 4, 2)
    stroke_set_encoder_summary = torchinfo.summary(stroke_set_encoder, input_data=[stroke_labels, control_points])

    print()
    print()

    image_size = 224
    images = torch.rand(batch_size, 1, image_size, image_size)  # (B, C, H, W)
    image_encoder_summary = torchinfo.summary(image_encoder, input_data=images)
