import torch
from torch import nn, optim
from torch.nn import functional as F
from kloader import KTensorDataLoader
from data import gen_mod_add
from models import HomoMLP
import argparse
import yaml
import math
import wandb

assert torch.cuda.is_available()

torch.set_default_device('cuda')

config = yaml.load(open('.config.yml'), Loader=yaml.FullLoader)

argparser = argparse.ArgumentParser()
argparser.add_argument('--data-seed', type=int, default=2435253235)
argparser.add_argument('--n-train', type=int, default=3763)
argparser.add_argument('--n-test', type=int, default=5646)
argparser.add_argument('--p', type=int, default=97)
argparser.add_argument('--h', type=int)
argparser.add_argument('--depth', type=int, default=2)
argparser.add_argument('--init-scale', type=float, default=1.0)
argparser.add_argument('--lr', type=float, default=0.1)
argparser.add_argument('--wd', type=float, default=1e-3)
argparser.add_argument('--only-train-last-layer', type=int, default=0)
argparser.add_argument('--last-layer-zero-init', type=int, default=0)
argparser.add_argument('--eval-first', type=int, default=1000)
argparser.add_argument('--eval-period', type=int, default=1000)
argparser.add_argument('--steps', type=int, default=100_000_000)
argparser.add_argument('--batch-size', type=int, default=10000)
argparser.add_argument('--eval-batch-size', type=int, default=10000)

args = argparser.parse_args()

wandb.init(
    project="grokking-mod-add",
    entity=config['wandb_entity'],
    name=f"N{args.n_train}-P{args.p}-H{args.h}-L{args.depth}-INIT{args.init_scale}-WD{args.wd}-OTLL{args.only_train_last_layer}-LLZI{args.last_layer_zero_init}",
    config=vars(args)
)
wandb.run.log_code(".")

train_data, test_data = gen_mod_add(args.data_seed, args.n_train, args.p)

train_loader = KTensorDataLoader(train_data, batch_size=min(args.batch_size, train_data[0].shape[0]), shuffle=True, drop_last=True)
train_loader_for_eval = KTensorDataLoader(train_data, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
test_loader = KTensorDataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

model = HomoMLP(init_scale=args.init_scale, L=args.depth, dimD=args.p * 2, dimH=args.h, dimO=args.p, first_layer_bias=True)
wandb.watch(model)

if args.last_layer_zero_init:
    model.layers[-1].weight.data.zero_()
if args.only_train_last_layer:
    for l in model.layers[:-1]:
        l.weight.requires_grad_(False)
        if l.bias is not None:
            l.bias.requires_grad_(False)

print('steps per epoch:', len(train_loader))

total_epochs = (args.steps + len(train_loader) - 1) // len(train_loader)

criterion = nn.CrossEntropyLoss()
optimier = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)


def separate_logits(y, target):
    N = y.shape[0]
    C = y.shape[1]
    correct = y[range(N), target]
    wrong = torch.masked_select(y, torch.logical_not(F.one_hot(target, num_classes=C))).reshape([N, C - 1])
    return correct, wrong


@torch.no_grad()
def eval_model(loader):
    acc = 0
    loss = 0
    margin = 10 ** 10
    n = 0
    for batch_x, batch_y in loader.iter():
        out = model(batch_x)
        n += batch_x.shape[0]
        loss += criterion(out, batch_y).item() * batch_x.shape[0]
        acc += (out.argmax(dim=-1) == batch_y).float().sum().item()
        
        c, w = separate_logits(out, batch_y)
        margin = min(margin, (c.view([-1, 1]) - w).min().item())

    return loss / n, acc / n, margin


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
continuous_time = 1
for eid in range(1, total_epochs):
    for bid, (batch_x, batch_y) in train_loader.enum():
        if cur_step % args.eval_period == 0 or cur_step <= args.eval_first:
            model.eval()

            log = {}
            train_loss, train_acc, train_margin = eval_model(train_loader_for_eval)
            log.update({
                'eval_train/loss': train_loss,
                'eval_train/acc': train_acc,
                'eval_train/margin': train_margin
            })
            test_loss, test_acc, test_margin = eval_model(test_loader)
            log.update({
                'eval_test/loss': test_loss,
                'eval_test/acc': test_acc,
                'eval_test/margin': test_margin
            })
            log.update(get_model_stats())
            log.update({
                'epoch': eid, 'train/step_in_epoch': bid, 'train/step': cur_step,
                'train/log_continuous_time': math.log(continuous_time),
                'train/continuous_time': continuous_time,
            })
            wandb.log(log)

            model.train()
        
        optimier.zero_grad(set_to_none=True)
        out = model(batch_x)
        loss = criterion(out, batch_y)
        
        continuous_time += args.lr
        
        loss.backward()
        optimier.step()

        cur_step += 1
