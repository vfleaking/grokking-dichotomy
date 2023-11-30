import torch
from torch import nn, optim
from torch.nn import functional as F
from kloader import KTensorDataLoader
from data import gen_matrix_for_completion
from models import MatrixFactorization
import argparse
import yaml
import wandb

assert torch.cuda.is_available()

torch.set_default_device('cuda')

config = yaml.load(open('.config.yml'), Loader=yaml.FullLoader)

argparser = argparse.ArgumentParser()
argparser.add_argument('--data-seed', type=int, default=35235125)
argparser.add_argument('--n-train', type=int)
argparser.add_argument('--d', type=int, default=97)
argparser.add_argument('--s', type=int, default=1)
argparser.add_argument('--alpha', type=float, default=1.0)
argparser.add_argument('--lr', type=float, default=0.1)
argparser.add_argument('--wd', type=float, default=1e-3)
argparser.add_argument('--eval-first', type=int, default=1000)
argparser.add_argument('--eval-period', type=int, default=1000)
argparser.add_argument('--steps', type=int, default=10_000_000)
argparser.add_argument('--batch-size', type=int, default=10000000)
argparser.add_argument('--eval-batch-size', type=int, default=1000)

args = argparser.parse_args()

wandb.init(
    project="grokking-mc",
    entity=config['wandb_entity'],
    name=f"N{args.n_train}-D{args.d}-S{args.s}-A{args.alpha}-WD{args.wd}",
    config=vars(args)
)
wandb.run.log_code(".")

M, train_data, test_data = gen_matrix_for_completion(args.data_seed, args.n_train, args.d, args.s, 'cuda')

train_loader = KTensorDataLoader(train_data, batch_size=min(args.batch_size, train_data[0].shape[0]), shuffle=True, drop_last=True)
train_loader_for_eval = KTensorDataLoader(train_data, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
test_loader = KTensorDataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

model = MatrixFactorization(alpha=args.alpha, dimD=args.d)
wandb.watch(model)

print('steps per epoch:', len(train_loader))

total_epochs = (args.steps + len(train_loader) - 1) // len(train_loader)

criterion = nn.MSELoss()
optimier = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)


@torch.no_grad()
def eval_model(loader):
    loss = 0
    n = 0
    for batch_x1, batch_x2, batch_y in loader.iter():
        out = model(batch_x1, batch_x2)
        n += batch_x1.shape[0]
        loss += criterion(out, batch_y).item() * batch_x1.shape[0]
    return loss / n


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
    for bid, (batch_x1, batch_x2, batch_y) in train_loader.enum():
        if cur_step % args.eval_period == 0 or cur_step <= args.eval_first:
            model.eval()

            log = {}
            train_loss = eval_model(train_loader_for_eval)
            log.update({ 'eval_train/loss': train_loss })
            test_loss = eval_model(test_loader)
            log.update({ 'eval_test/loss': test_loss })
            log.update(get_model_stats())
            log.update({ 'epoch': eid, 'train/step_in_epoch': bid, 'train/step': cur_step })
            wandb.log(log)

            model.train()
        
        optimier.zero_grad(set_to_none=True)
        out = model(batch_x1, batch_x2)
        loss = criterion(out, batch_y)
        loss.backward()
        optimier.step()

        cur_step += 1
