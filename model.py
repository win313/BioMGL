import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class ACRCAM(nn.Module):
    """ACR-CAM = agreement-weighted consensus + MAFN-consistent residual CAM."""

    def __init__(
        self,
        hidden_dim,
        acr_norm="none",
        acr_coeff_mode="scalar",
        acr_debug=False,
        eps=1e-8,
        balanced_att_init=False,
        avg_fusion_init=False,
        acr_tau=1.0,
    ):
        super(ACRCAM, self).__init__()
        if acr_norm not in ("none", "rownorm_agree", "rownorm_all", "layernorm"):
            raise ValueError("unknown acr_norm: {}".format(acr_norm))
        if acr_coeff_mode not in ("scalar", "featurewise"):
            raise ValueError("unknown acr_coeff_mode: {}".format(acr_coeff_mode))
        if acr_tau <= 0:
            raise ValueError("acr_tau must be positive")
        self.hidden_dim = hidden_dim
        self.acr_norm = acr_norm
        self.acr_coeff_mode = acr_coeff_mode
        self.acr_debug = acr_debug
        self.eps = eps
        self.acr_tau = acr_tau
        self._debug_printed = False

        self.att_fc = nn.Linear(3 * hidden_dim, 3)
        self.feature_att_fc = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_fc = nn.Linear(3 * hidden_dim, hidden_dim)
        if acr_norm == "layernorm":
            self.ln_spa = nn.LayerNorm(hidden_dim)
            self.ln_exp = nn.LayerNorm(hidden_dim)
            self.ln_sem = nn.LayerNorm(hidden_dim)
        if balanced_att_init:
            self._init_balanced_feature_att()
        if avg_fusion_init:
            self._init_average_fusion_fc()

    def _init_balanced_feature_att(self):
        # Initialize feature-wise CAM with equal coefficients for spa, exp, sem.
        nn.init.zeros_(self.feature_att_fc.weight)
        nn.init.ones_(self.feature_att_fc.bias)

    def _init_average_fusion_fc(self):
        # Initialize residual fusion as one-third identity for each residual block.
        with torch.no_grad():
            self.fusion_fc.weight.zero_()
            self.fusion_fc.bias.zero_()

            d = self.hidden_dim
            eye = torch.eye(d)
            self.fusion_fc.weight[:, 0:d].copy_(eye / 3.0)
            self.fusion_fc.weight[:, d:2 * d].copy_(eye / 3.0)
            self.fusion_fc.weight[:, 2 * d:3 * d].copy_(eye / 3.0)
    def row_cos(self, a, b):
        return F.cosine_similarity(a, b, dim=1, eps=self.eps).unsqueeze(1)

    def normalize_scalar_cam_coeff(self, u_raw):
        return F.normalize(u_raw, p=2, dim=1, eps=self.eps)

    def normalize_featurewise_cam_coeff(self, coeff):
        return F.normalize(coeff, p=2, dim=1, eps=self.eps)

    def _prepare_views(self, h_spa, h_exp, h_sem):
        if self.acr_norm == "layernorm":
            h_spa = self.ln_spa(h_spa)
            h_exp = self.ln_exp(h_exp)
            h_sem = self.ln_sem(h_sem)
            return h_spa, h_exp, h_sem, h_spa, h_exp, h_sem

        if self.acr_norm == "rownorm_all":
            h_spa = F.normalize(h_spa, p=2, dim=1, eps=self.eps)
            h_exp = F.normalize(h_exp, p=2, dim=1, eps=self.eps)
            h_sem = F.normalize(h_sem, p=2, dim=1, eps=self.eps)
            return h_spa, h_exp, h_sem, h_spa, h_exp, h_sem

        if self.acr_norm == "rownorm_agree":
            h_spa_agree = F.normalize(h_spa, p=2, dim=1, eps=self.eps)
            h_exp_agree = F.normalize(h_exp, p=2, dim=1, eps=self.eps)
            h_sem_agree = F.normalize(h_sem, p=2, dim=1, eps=self.eps)
            return h_spa_agree, h_exp_agree, h_sem_agree, h_spa, h_exp, h_sem

        return h_spa, h_exp, h_sem, h_spa, h_exp, h_sem

    def forward(self, h_spa, h_exp, h_sem):
        h_spa_agree, h_exp_agree, h_sem_agree, h_spa_base, h_exp_base, h_sem_base = self._prepare_views(
            h_spa, h_exp, h_sem
        )

        g_spa = 0.5 * (
            self.row_cos(h_spa_agree, h_exp_agree)
            + self.row_cos(h_spa_agree, h_sem_agree)
        )
        g_exp = 0.5 * (
            self.row_cos(h_exp_agree, h_spa_agree)
            + self.row_cos(h_exp_agree, h_sem_agree)
        )
        g_sem = 0.5 * (
            self.row_cos(h_sem_agree, h_spa_agree)
            + self.row_cos(h_sem_agree, h_exp_agree)
        )
        agreement = torch.cat([g_spa, g_exp, g_sem], dim=1)
        pi = torch.softmax(agreement / self.acr_tau, dim=1)

        consensus = (
            pi[:, 0:1] * h_spa_base
            + pi[:, 1:2] * h_exp_base
            + pi[:, 2:3] * h_sem_base
        )

        delta_spa = h_spa_base - consensus
        delta_exp = h_exp_base - consensus
        delta_sem = h_sem_base - consensus

        residual = torch.cat([delta_spa, delta_exp, delta_sem], dim=1)
        u_raw = None
        u = None

        if self.acr_coeff_mode == "scalar":
            u_raw = self.att_fc(residual)
            u = self.normalize_scalar_cam_coeff(u_raw)
            weighted_residual = torch.cat([
                u[:, 0:1] * delta_spa,
                u[:, 1:2] * delta_exp,
                u[:, 2:3] * delta_sem,
            ], dim=1)
        else:
            # Feature-wise mode keeps a consistent view order: spa, exp, sem.
            residual_stack = torch.stack([delta_spa, delta_exp, delta_sem], dim=1)
            u_raw = self.feature_att_fc(residual_stack)
            u = self.normalize_featurewise_cam_coeff(u_raw)
            weighted_residual = torch.cat([
                u[:, 0, :] * delta_spa,
                u[:, 1, :] * delta_exp,
                u[:, 2, :] * delta_sem,
            ], dim=1)

        residual_out = self.fusion_fc(weighted_residual)
        z = consensus + residual_out

        if self.acr_debug and not self._debug_printed:
            print("ACR-CAM debug:")
            print("  H_spa shape:", tuple(h_spa.shape))
            print("  H_exp shape:", tuple(h_exp.shape))
            print("  H_sem shape:", tuple(h_sem.shape))
            print("  G shape:", tuple(agreement.shape))
            print("  Pi shape:", tuple(pi.shape), "row_sum_mean:", float(pi.sum(dim=1).mean().detach().cpu()))
            print("  C shape:", tuple(consensus.shape))
            print("  acr_coeff_mode:", self.acr_coeff_mode)
            print("  U_raw shape:", tuple(u_raw.shape))
            print("  U shape:", tuple(u.shape))
            print("  Weighted_R shape:", tuple(weighted_residual.shape))
            print("  Z shape:", tuple(z.shape))
            self._debug_printed = True

        return z


class MAFN_ACRCAM(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid1,
        nhid2,
        dropout,
        acr_norm="none",
        acr_coeff_mode="scalar",
        acr_debug=False,
        balanced_att_init=False,
        avg_fusion_init=False,
        acr_tau=1.0,
    ):
        super(MAFN_ACRCAM, self).__init__()

        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ACR_CAM = ACRCAM(
            nhid2,
            acr_norm=acr_norm,
            acr_coeff_mode=acr_coeff_mode,
            acr_debug=acr_debug,
            balanced_att_init=balanced_att_init,
            avg_fusion_init=avg_fusion_init,
            acr_tau=acr_tau,
        )
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout

    def forward(self, x, sadj, fadj, semadj):
        emb_spa = self.SGCN(x, sadj)
        emb_exp = self.FGCN(x, fadj)
        emb_sem = self.CGCN(x, semadj)

        emb = self.ACR_CAM(emb_spa, emb_exp, emb_sem)
        [pi, disp, mean] = self.ZINB(emb)
        return emb, pi, disp, mean, emb_spa, emb_exp, emb_sem


def build_model(
    fusion_type,
    nfeat,
    nhid1,
    nhid2,
    dropout,
    acr_norm="none",
    acr_coeff_mode="scalar",
    acr_debug=False,
    balanced_att_init=False,
    avg_fusion_init=False,
    acr_tau=1.0,
):
    if fusion_type == "acr_cam":
        return MAFN_ACRCAM(
            nfeat=nfeat,
            nhid1=nhid1,
            nhid2=nhid2,
            dropout=dropout,
            acr_norm=acr_norm,
            acr_coeff_mode=acr_coeff_mode,
            acr_debug=acr_debug,
            balanced_att_init=balanced_att_init,
            avg_fusion_init=avg_fusion_init,
            acr_tau=acr_tau,
        )
    raise ValueError("unknown fusion_type: {}".format(fusion_type))
