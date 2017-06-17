import argparse
import matplotlib.pyplot as plt
import numpy as np

from pixel2torque.pytorch.e2c import E2C
from pixel2torque.pytorch.vae import VAE, latent_loss
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pixel2torque.tf_e2c.plane_data2 import PlaneData
from pixel2torque.pytorch.datasets import PlaneDataset


def show_and_save(img, path):
    npimg = img.numpy()
    plt.imsave(path, np.transpose(npimg, (1, 2, 0)))


def weights_init(m):
    from torch.nn.init import xavier_uniform
    if isinstance(m, torch.nn.Linear):
        xavier_uniform(m.weight.data)


def compute_loss(reconst, actual, mean, var):
    ll = latent_loss(mean, var)
    reconst_loss = torch.sum(reconst.sub(actual).pow(2), dim=1)
    loss = reconst_loss.add(ll).mean()
    return loss


def parse_args():
    parser = argparse.ArgumentParser(description='train e2c on plane data')
    parser.add_argument('--batch-size', required=False, default=128, type=int)
    parser.add_argument('--lr', required=False, default=3e-4, type=float)
    parser.add_argument('--seed', required=False, default=1234, type=int)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--test-frequency', default=100)
    parser.add_argument('--eval-embedding-frequency', default=200)
    parser.add_argument('--visual', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    img_width = 40
    width = img_width
    height = width
    model_dir = args.model_dir

    planedata = PlaneData('plane0.npz', 'env0.png')
    planedata.initialize()

    dataset = PlaneDataset(planedata)
    trainset = dataset
    testset = dataset

    batch_size = args.batch_size

    print('batch_size %d' % batch_size)

    loader = DataLoader(trainset, batch_size, shuffle=False,
                        drop_last=False)

    input_dim = width * height
    latent_dim = 2
    action_dim = 2

    model = E2C(input_dim, latent_dim, action_dim)

    print(model)
    weights_init(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.1, 0.1))

    aggregate_loss = None
    latent_loss = None
    reconst_loss = None

    if args.visual:
        fig = plt.figure('Embeddings')
        plt.ion()
        true_ax = plt.subplot(121)
        pred_ax = plt.subplot(122)


    def evaluate_embedding(model):
        ps = planedata.getPSpace()
        xs = []
        for p in ps:
            xs.append(np.array(planedata.getXp(p.astype(np.int8))))
        xs = Variable(torch.from_numpy(np.array(xs)).view(len(ps), input_dim))
        embeds = model.latent_embeddings(xs)

        xs = embeds.data.numpy()[:, 0]
        ys = embeds.data.numpy()[:, 1]
        colors = np.array([i for i in range(len(xs))]) / len(xs)

        if args.visual:
            true_ax.scatter(ps[:, 0], ps[:, 1], c=colors)
            pred_ax.scatter(xs, ys, c=colors)
            plt.pause(0.01)

        return embeds


    for epoch in range(1000):
        if args.visual:
            fig.suptitle('Epoch: {}'.format(epoch))
        model.train()
        for i, (x, u, x_next) in enumerate(loader):
            x = Variable(
                x.resize_(batch_size, input_dim).float(), requires_grad=False)
            u = Variable(u.float())
            x_next = Variable(
                x_next.resize_(batch_size, input_dim).float(),
                requires_grad=False)

            optimizer.zero_grad()

            dec, loss = model(x, u, x_next)
            loss = loss()
            loss.backward()
            optimizer.step()

            aggregate_loss = loss.data[0]

        if epoch % args.eval_embedding_frequency == 0:
            embeds = evaluate_embedding(model)

        if epoch % args.test_frequency == 0:
            checkpoint_filename = 'checkpoint.pth'
            torch.save(model, checkpoint_filename)
            print('[Epoch %d] checkpoint saved to %s' % (epoch, checkpoint_filename))
            print('[Epoch %d] loss: %f' % (epoch, aggregate_loss))
