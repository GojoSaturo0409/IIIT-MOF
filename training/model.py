import torch
import torch.nn as nn

def patchify_batch(x, patch):
    B, C, G, _, _ = x.shape
    assert G % patch == 0
    n = G // patch
    x = x.view(B, C, n, patch, n, patch, n, patch)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    patches = x.view(B, n*n*n, C * (patch ** 3))
    return patches

def random_masking(N, ratio, device):
    N_mask = int(N * ratio)
    perm = torch.randperm(N, device=device)
    mask_indices, _ = torch.sort(perm[:N_mask])
    keep_indices, _ = torch.sort(perm[N_mask:])
    return mask_indices.long(), keep_indices.long()

def mae_loss_on_masked(patches, pred, mask_indices):
    target = patches.index_select(1, mask_indices)
    pred_m = pred.index_select(1, mask_indices)
    return ((pred_m - target) ** 2).mean()

class PatchEmbed(nn.Module):
    def __init__(self, patch_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
    def forward(self, x): return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(embed_dim, heads, 4*embed_dim, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
    def forward(self, x): return self.enc(x)

class MAE3D(nn.Module):
    def __init__(self, patch_dim, enc_embed, enc_depth, enc_heads,
                 dec_embed, dec_depth, dec_heads, mask_ratio):
        super().__init__()
        self.enc_embed = enc_embed
        self.patch_embed = PatchEmbed(patch_dim, enc_embed)
        self.encoder = Transformer(enc_embed, enc_depth, enc_heads)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed))
        self.enc_to_dec = nn.Linear(enc_embed, dec_embed)
        self.pos_embed_enc = None
        self.pos_embed_dec = None
        self.decoder = Transformer(dec_embed, dec_depth, dec_heads)
        self.dec_to_patch = nn.Linear(dec_embed, patch_dim)
        self.mask_ratio = mask_ratio
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        nn.init.xavier_uniform_(self.enc_to_dec.weight)
        nn.init.xavier_uniform_(self.dec_to_patch.weight)
        nn.init.normal_(self.mask_token, std=0.02)

    def init_pos_embeds(self, N, device):
        if self.pos_embed_enc is None or self.pos_embed_enc.device != device:
            self.pos_embed_enc = nn.Parameter(torch.zeros(1, N, self.enc_embed, device=device))
            nn.init.normal_(self.pos_embed_enc, std=0.02)
        if self.pos_embed_dec is None or self.pos_embed_dec.device != device:
            self.pos_embed_dec = nn.Parameter(torch.zeros(1, N, self.enc_to_dec.out_features, device=device))
            nn.init.normal_(self.pos_embed_dec, std=0.02)

    def encode(self, patches):
        x = self.patch_embed(patches)
        if self.pos_embed_enc is not None:
            x = x + self.pos_embed_enc[:, :x.shape[1], :]
        return self.encoder(x).mean(dim=1)

    def forward(self, patches, mask_indices, keep_indices):
        B, N, D = patches.shape
        self.init_pos_embeds(N, patches.device)

        x_vis = self.patch_embed(patches.index_select(1, keep_indices))
        x_vis += self.pos_embed_enc.index_select(1, keep_indices)
        
        enc_out = self.encoder(x_vis)
        dec_vis = self.enc_to_dec(enc_out)
        
        dec_tokens = self.mask_token.expand(B, N, -1).clone()
        dec_tokens.scatter_(1, keep_indices.unsqueeze(-1).expand(-1, -1, dec_tokens.shape[-1]), dec_vis)
        dec_tokens += self.pos_embed_dec
        
        return self.dec_to_patch(self.decoder(dec_tokens))
