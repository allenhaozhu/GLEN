# NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
import torch
from utils import load_adj_scatter, load_dataset_adj_lap
from mlp import Single
import argparse
import numpy as np
from classification import classify


def compute_discrimn_loss_theoretical(W, eps=0.001):
    """Theoretical Discriminative Loss."""
    m, p = W.shape
    I = torch.eye(p)
    scalar = p / (m * eps)
    logdet = torch.logdet(I + scalar * W.T.matmul(W))
    return logdet / 2.


def compute_compress_loss_theoretical(W, P, eps=0.01):
    """Theoretical Compressive Loss."""
    m, p = W.shape
    #k, _, _ = Pi.shape
    I = torch.eye(p)
    # compress_loss = 0.
    # trPi = torch.trace(P) + 1e-8
    # scalar = p / (trPi * eps)
    log_det = torch.logdet(I + eps * W.T.matmul(P).matmul(W))
    compress_loss = 0.5*log_det
    return compress_loss

def computer_decorelation_loss(W):

    corr = torch.einsum("bi, bj -> ij", W, W) / W.shape[0]
    diag = torch.eye(W.shape[1], device=corr.device)
    #cdif = (diag.mul(corr) - corr).pow(2)
    cdif = (diag - corr).pow(2)
    return cdif.sum()

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='citeseer',
                    help='dataset')
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--nhid', type=int, default=1024,
                    help='hidden size')
parser.add_argument('--output', type=int, default=512,
                    help='output size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=8,
                    help='maximum number of epochs')
parser.add_argument('--sample', type=int, default=5,
                    help='    ')
parser.add_argument('--num_nodes', type=int, default=3327,
                    help='    ')
parser.add_argument('--num_features', type=int, default=3703,
                    help='    ')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature, adj_normalized, lap_normalized= load_dataset_adj_lap(args.dataset)
feature = feature.to(device)
#feature = torch.nn.functional.normalize(feature, dim=1)
adj_normalized = adj_normalized.to(device)
lap_normalized = lap_normalized.to(device)
K = 8
emb = feature.clone()
for i in range(K):
    feature = torch.mm(adj_normalized, feature)
    emb = emb + feature
emb/=K
#neg_sample = torch.from_numpy(load_adj_neg(args.num_nodes, args.sample)).float().to(device)
adj_scatter = torch.from_numpy(load_adj_scatter(args.num_nodes)).float().to(device)
model = Single(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()
#I = torch.eye(args.output)
#bn = torch.nn.BatchNorm1d(args.output, affine=False)
for epoch in range(args.epochs):

    optimizer.zero_grad()
    #out = bn(model(emb))
    out = model(emb)
    # loss = (compute_compress_loss_theoretical(out, lap_normalized, 4) - compute_discrimn_loss_theoretical(out,
    #                                                                                                           4)) + 2 * computer_decorelation_loss(
    #     out)
    import math
    alpha = 0.018
    loss = math.log(1+alpha)*(compute_compress_loss_theoretical(out,lap_normalized, alpha) - compute_compress_loss_theoretical(out,adj_scatter, alpha))#+#compute_discrimn_loss_theoretical(out, 16)# + 4*computer_decorelation_loss(out)
    #loss = compute_compress_loss_theoretical(out, adj_normalized, 2) - compute_discrimn_loss_theoretical(out,2) + 4*computer_decorelation_loss(out)
    #loss =  8*computer_decorelation_loss(out)
    #loss = - compute_discrimn_loss_theoretical(out, 32)
    print(loss)
    loss.backward()
    optimizer.step()

model.eval()
emb = model(emb).cpu().detach().numpy()
np.save('embedding.npy', emb)
classify(emb, args.dataset, per_class='20')
classify(emb, args.dataset, per_class='5')
# 75.94010614101592
# 2.558565685178548
# 81.08649093904448
# 1.2309989030251056
