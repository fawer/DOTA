import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from torch.autograd import Variable
import matplotlib.pyplot as plt
import ot
from geomloss import SamplesLoss
import cvae
from sklearn.metrics import roc_auc_score


def Guassian_loss(recon_x, x):
    weights = x * args.alpha + (1 - x)
    loss = x - recon_x
    loss = torch.sum(weights * loss * loss)
    return loss


def BCE_loss(recon_x, x):
    eps = 1e-8
    loss = -torch.sum(args.alpha * torch.log(recon_x + eps) * x + torch.log(1 - recon_x + eps) * (1 - x))
    return loss


# cpu version
class WaLoss(nn.Module):
    def __init__(self):
        super(WaLoss, self).__init__()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs, target):
        inputs = torch.squeeze(inputs, 0)
        loss = torch.zeros(inputs.shape).to(args.device)
        mse = (inputs - target) ** 2
        n = inputs.shape[0]
        x = torch.arange(n, dtype=float)
        M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
        M = M / M.max()
        Gs = torch.from_numpy(ot.sinkhorn(inputs.cpu().detach().numpy(), target.cpu(), M, reg=0.2))
        Gs = self.softmax(Gs)
        loss += (Gs * mse)
        return loss.mean()


def train(epoch, train_loader, loss_function, optimizer, model):
    model.train()
    loss_value = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(args.device)
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        recon_batch = recon_batch.squeeze(0)
        loss = args.alpha * loss_function(recon_batch, data) + \
            cvae.regularization(mu, log_var) * args.beta + 0.1 * BCE_loss(recon_batch, data)
        loss.backward()
        loss_value += loss.item()
        optimizer.step()
        # if args.log != 0 and batch_idx % args.log == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader),
        #                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, loss_value / len(train_loader.dataset)))
    return loss_value / len(train_loader.dataset)


def evaluate(targets, inputs, mod):
    targets = targets.cpu().detach().numpy()
    targets = np.where(targets == 2, 1, targets).reshape(1, -1)
    temp_y, _, _ = mod(inputs)
    temp_y = temp_y.cpu().detach().numpy().reshape(1, -1)
    # roc and prc
    fpr_s, tpr_s, _ = roc_curve(targets[0], temp_y[0], pos_label=1)
    precisions, recalls, thresholds = precision_recall_curve(targets[0], temp_y[0], pos_label=1)
    return fpr_s, tpr_s, precisions, recalls, roc_auc_score(targets[0], temp_y[0]), temp_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--batch', type=int, default=300, help='input batch size for training (default: 100)')
    parser.add_argument('-m', '--maxiter', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dir', help='dataset directory', default='/Users/deepDR/dataset')
    parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[20])
    parser.add_argument('-L', type=int, default=1, help='number of samples')
    parser.add_argument('-N', help='number of recommended items', type=int, default=20)
    parser.add_argument('--learn_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=1)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=1)
    parser.add_argument('--rating', help='feed input as rating', action='store_true')
    parser.add_argument('--save', help='save model', action='store_true')
    parser.add_argument('--fig', help='print figure', action='store_true')
    parser.add_argument('--load', help='load model, 1 for fvae and 2 for cvae', type=int, default=0)
    parser.add_argument('--dropout', help='dropout rate', type=int, default=0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # whether to ran with cuda
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(args.device)
    directory = args.dir
    ad_tensor = np.load("./dataset/new_dd.npy").astype(np.float32)
    ad_tensor = torch.from_numpy(ad_tensor).to(args.device)
    tensor = np.load('./dataset/new_dd_al.npy').astype(np.float32)
    tensor = torch.from_numpy(tensor).to(args.device)
    rs = np.random.randint(0, 1000, 1)[0]
    # cross validate
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    cv_count = 0
    train_fp, test_fp, train_tp, test_tp = [], [], [], []
    train_p, test_p, train_r, test_r = [], [], [], []
    train_rs, test_rs = [], []
    # define model, data, loss optimizer
    args.d = tensor.shape[1]
    dummy = torch.zeros(tensor.shape[0])
    model = cvae.VAE(args).to(args.device)
    # pre-train with drug feature
    if not args.rating:
        path = 'drugmdaFeatures.txt'
        fea = np.loadtxt(path)
        X = fea.transpose()
        X[X > 0] = 1
        args.d = X.shape[1]
        # X = normalize(X, axis=1)
        X = torch.from_numpy(X.astype('float32')).to(args.device)
        train_loader = DataLoader(X, args.batch, shuffle=True)
        loss_function = Guassian_loss
        optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

        for epochs in range(1, args.maxiter + 1):
            train(epochs, train_loader, loss_function, optimizer, model)

    # train with drug-disease relation
    else:
        for train_id, test_id in kf.split(tensor, dummy):
            train_x, train_y = tensor[train_id], tensor[train_id]
            test_x, test_y = tensor[test_id], tensor[test_id]
            train_loader = DataLoader(train_x, args.batch, shuffle=True)
            # define loss function
            loss_function = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            # loss_function = BCE_loss

            # load model
            if args.load > 0:
                name = 'cvae' if args.load == 2 else 'fvae'
                path = 'test_models/' + name
                for l in args.layer:
                    path += '_' + str(l)
                print('load model from path: ' + path)
                model.load_state_dict(torch.load(path))

            optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

            # training
            loss_list = []
            for epochs in range(1, args.maxiter + 1):
                loss = train(epochs, train_loader, loss_function, optimizer, model)
                loss_list.append(loss)

            model.eval()
            # get training set results
            fpr, tpr, precision, recall, auc_score, _ = evaluate(train_y, train_x, model)
            train_rs.append(auc_score)
            train_fp.append(fpr)
            train_tp.append(tpr)
            train_p.append(precision)
            train_r.append(recall)

            # get testing results
            fpr, tpr, precision, recall, auc_score, _ = evaluate(test_y, test_x, model)
            test_rs.append(auc_score)
            test_fp.append(fpr)
            test_tp.append(tpr)
            test_p.append(precision)
            test_r.append(recall)
            score, _, _ = model(ad_tensor)
            score = score.cpu().detach().numpy()
            np.save(f'predict/score_wl_{cv_count}_oaz.npy', score)
            cv_count += 1
        # get figure
        if args.fig:
            plt.subplot(2, 2, 1)
            for i in range(5):
                plt.plot(train_fp[i], train_tp[i])
            plt.title('ROC of training set')
            plt.subplot(2, 2, 2)
            for i in range(5):
                plt.plot(test_fp[i], test_tp[i])
            plt.title('ROC of testing set')
            plt.subplot(2, 2, 3)
            for i in range(5):
                plt.plot(train_r[i], train_p[i])
            plt.title('PRC of training set')
            plt.subplot(2, 2, 4)
            for i in range(5):
                plt.plot(test_r[i], test_p[i])
            plt.title('PRC of testing set')
            print(f'train auc:{sum(train_rs)/len(train_rs)} test auc: {sum(test_rs)/len(test_rs)}')
            plt.show()
        # save model
        if args.save:
            name = 'cvae' if args.rating else 'fvae'
            path = 'test_models/' + name
            for l in args.layer:
                path += '_' + str(l)
            model.cpu()
            torch.save(model.state_dict(), path)