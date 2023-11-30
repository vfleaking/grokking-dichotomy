import torch
from torch import nn, optim
from torch.nn import functional as F
from kloader import KTensorDataLoader
from data import gen_l2_separated_linear_classification
from models import DiagonalNet
import argparse
import yaml
import wandb

assert torch.cuda.is_available()

torch.set_default_device('cuda')

config = yaml.load(open('.config.yml'), Loader=yaml.FullLoader)

argparser = argparse.ArgumentParser()
argparser.add_argument('--data-seed', type=int, default=24181325235)
argparser.add_argument('--n-train', type=int)
argparser.add_argument('--n-test', type=int, default=10000)
argparser.add_argument('--d', type=int, default=100000)
argparser.add_argument('--gamma', type=float, default=25)
argparser.add_argument('--L', type=int, default=2)
argparser.add_argument('--alpha', type=float, default=1.0)
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--wd', type=float, default=1e-3)
argparser.add_argument('--eval-first', type=int, default=1000)
argparser.add_argument('--eval-period', type=int, default=1000)
argparser.add_argument('--steps', type=int, default=1_000_000)
argparser.add_argument('--batch-size', type=int, default=10000000)
argparser.add_argument('--eval-batch-size', type=int, default=1000)

args = argparser.parse_args()

wandb.init(
    project="grokking-diag-cls2",
    entity=config['wandb_entity'],
    name=f"N{args.n_train}-D{args.d}-G{args.gamma}-L{args.L}-A{args.alpha}-WD{args.wd}",
    config=vars(args)
)
wandb.run.log_code(".")

train_data, test_data = gen_l2_separated_linear_classification(args.data_seed, args.n_train, args.n_test, args.d, args.gamma, 'cuda')

train_loader = KTensorDataLoader(train_data, batch_size=min(args.batch_size, train_data[0].shape[0]), shuffle=True, drop_last=True)
train_loader_for_eval = KTensorDataLoader(train_data, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
test_loader = KTensorDataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

model = DiagonalNet(alpha=args.alpha, L=args.L, dimD=args.d)
wandb.watch(model)

print('steps per epoch:', len(train_loader))

total_epochs = (args.steps + len(train_loader) - 1) // len(train_loader)

def logistic_loss(out, y):
    return F.softplus(-out * y).mean()

criterion = logistic_loss
optimier = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)


@torch.no_grad()
def eval_model(loader):
    acc = 0
    loss = 0
    n = 0
    for batch_x, batch_y in loader.iter():
        out = model(batch_x)[:, 0]
        n += batch_x.shape[0]
        loss += criterion(out, batch_y).item() * batch_x.shape[0]
        val = out * batch_y
        acc += ((val > 0) + (val == 0) / 2).sum().item()
    return loss / n, acc / n


@torch.no_grad()
def get_model_stats():
    stats = {}
    total_norm2 = 0
    for name, param in model.named_parameters():
        cur_norm2 = (param ** 2).sum().item()
        stats[f'norm/{name}'] = cur_norm2 ** 0.5
        total_norm2 += cur_norm2
    stats[f'total_norm'] = total_norm2 ** 0.5
    return stats

model.train()

cur_step = 0
for eid in range(1, total_epochs):
    for bid, (batch_x, batch_y) in train_loader.enum():
        if cur_step % args.eval_period == 0 or cur_step <= args.eval_first:
            model.eval()

            log = {}
            train_loss, train_acc = eval_model(train_loader_for_eval)
            log.update({ 'eval_train/loss': train_loss, 'eval_train/acc': train_acc })
            test_loss, test_acc = eval_model(test_loader)
            log.update({ 'eval_test/loss': test_loss, 'eval_test/acc': test_acc })
            log.update(get_model_stats())
            log.update({ 'epoch': eid, 'train/step_in_epoch': bid, 'train/step': cur_step })
            wandb.log(log)

            model.train()
        
        optimier.zero_grad(set_to_none=True)
        out = model(batch_x)[:, 0]
        loss = criterion(out, batch_y)
        loss.backward()
        optimier.step()

        cur_step += 1
