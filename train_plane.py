import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pixel2torque.pytorch.e2c import E2C
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pixel2torque.tf_e2c.plane_data2 import PlaneData
from pixel2torque.pytorch.datasets import PlaneDataset

USE_CUDA = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser(description='train e2c on plane data')
    parser.add_argument('--batch-size', required=False, default=128, type=int)
    parser.add_argument('--lr', required=False, default=1e-4, type=float)
    parser.add_argument('--seed', required=False, default=0, type=int)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--test-frequency', default=100)
    parser.add_argument('--eval-embedding-frequency', default=100)
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--lambd', default=0.25)
    parser.add_argument('--continue-from', default=None)
    return parser.parse_args()


def weights_init(m):
    from torch.nn.init import xavier_uniform, orthogonal
    if isinstance(m, torch.nn.Linear):
        orthogonal(m.weight.data, 1.1)


def make_model_dir(basedir, params):
    hyperparams = ['lr', 'lambd']
    return os.path.join(basedir, *['-'.join([p, str(params[p])]) for p in hyperparams])


def checkpoint(epoch, model, model_dir):
    path = os.path.join(model_dir, 'model.pth-{:05d}'.format(epoch))
    torch.save(model, path)
    print('checkpoint saved to {}'.format(path))


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if USE_CUDA:
        pass

    img_width = 40
    width = img_width
    height = width

    input_dim = width * height
    latent_dim = 2
    action_dim = 2
    batch_size = args.batch_size

    hyperparams = {'lr': args.lr, 'lambd': args.lambd}

    model_dir = make_model_dir(args.model_dir, hyperparams)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=False)

    planedata = PlaneData('plane0.npz', 'env1.png')
    planedata.initialize()

    dataset = PlaneDataset(planedata)

    loader = DataLoader(dataset, batch_size, shuffle=True,
                        drop_last=False, num_workers=4)

    model = E2C(input_dim, latent_dim, action_dim, config='plane')

    print('batch_size %d' % batch_size)
    print(model)
    weights_init(model)

    if args.continue_from:
        print('Continue training from %s' % args.continue_from)
        model = torch.load(args.continue_from)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.1, 0.1))

    aggregate_loss = None
    latent_loss = None
    reconst_loss = None

    if args.visual:
        plt.ion()
        fig = plt.figure('Embeddings')
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
            true_ax.clear()
            pred_ax.clear()
            true_ax.scatter(ps[:, 0], ps[:, 1], c=colors)
            pred_ax.scatter(xs, ys, c=colors)

        return embeds

    step = 0
    for epoch in range(args.epochs):
        model.train()
        for i, (x, u, x_next) in enumerate(loader):
            x = Variable(
                x.resize_(batch_size, input_dim).float(), requires_grad=False)
            u = Variable(u.float())
            x_next = Variable(
                x_next.resize_(batch_size, input_dim).float(),
                requires_grad=False)

            if USE_CUDA:
                x.cuda()
                u.cuda()
                x_next.cuda()

            optimizer.zero_grad()

            dec, closure = model(x, u, x_next)
            bound_loss, kl_loss = closure()
            loss = bound_loss.add(kl_loss)
            loss.backward()
            optimizer.step()

            aggregate_loss = loss.data[0]

            if step % 1000 == 0:
                if args.visual:
                    fig.suptitle('Step: {}'.format(step))
                embeds = evaluate_embedding(model)
                plt.savefig(os.path.join(model_dir, 'embeddings_step-{:05d}'.format(step)))
                plt.pause(0.01)

            if step >= 10e5:
                sys.exit(0)

            if step % 1000 == 0:
                checkpoint(step, model, model_dir)

            if step % 1000 == 0:
                print('step: {}, loss: {}, bound_loss: {}, kl_loss: {}' \
                      .format(step, aggregate_loss, bound_loss.data[0], kl_loss.data[0]))

            step += 1

            # if epoch % args.eval_embedding_frequency == 0:
            #     embeds = evaluate_embedding(model)
            #
            # if epoch % args.test_frequency == 0:
            #     checkpoint(epoch, model, model_dir)
            #     print('[Epoch %d] loss: %f' % (epoch, aggregate_loss))
