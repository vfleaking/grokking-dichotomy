import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

def gen_xor(seed, n_train, n_test, d, s):
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    train_inputs = torch.randint(0, 2, (n_train, d), dtype=torch.float32) * 2 - 1
    test_inputs = torch.randint(0, 2, (n_test, d), dtype=torch.float32) * 2 - 1

    train_targets = torch.ones([n_train])
    test_targets = torch.ones([n_test])
    for i in range(s):
        train_targets *= train_inputs[:, i]
        test_targets *= test_inputs[:, i]

    torch.set_rng_state(old_state)

    return (train_inputs, train_targets), (test_inputs, test_targets)

def gen_sparse_linear_classification(seed, n_train, n_test, d, s, device):
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    train_inputs = torch.randint(0, 2, (n_train, d), dtype=torch.float32, device=device) * 2 - 1
    test_inputs = torch.randint(0, 2, (n_test, d), dtype=torch.float32, device=device) * 2 - 1
    
    w = torch.randint(0, 2, (s,), dtype=torch.float32, device=device) * 2 - 1

    train_targets = (train_inputs[:, :s] @ w > 0) * 2 - 1
    test_targets = (test_inputs[:, :s] @ w > 0) * 2 - 1

    torch.set_rng_state(old_state)

    return (train_inputs, train_targets), (test_inputs, test_targets)
    
def gen_l2_separated_linear_classification(seed, n_train, n_test, d, gamma, device):
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    w = torch.randn(d, device=device)
    w = w / torch.linalg.norm(w)

    train_inputs = torch.randn((n_train, d), dtype=torch.float32, device=device)
    test_inputs = torch.randn((n_test, d), dtype=torch.float32, device=device)

    train_targets = (train_inputs @ w > 0) * 2 - 1
    test_targets = (test_inputs @ w > 0) * 2 - 1

    train_inputs = train_inputs + train_targets.unsqueeze(-1) * (gamma / 2) * w
    test_inputs = test_inputs + test_targets.unsqueeze(-1) * (gamma / 2) * w

    torch.set_rng_state(old_state)

    return (train_inputs, train_targets), (test_inputs, test_targets)

def gen_matrix_for_completion(seed, n_train, d, s, device):
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    assert s == 1

    v = torch.arange(0, d).unsqueeze(-1) / d
    M = v @ v.T
    
    perm = torch.randperm(d * d, device=device)
    
    train_x1 = perm[:n_train] // d
    train_x2 = perm[:n_train] % d
    train_targets = M[train_x1, train_x2]
    
    test_x1 = perm[n_train:] // d
    test_x2 = perm[n_train:] % d
    test_targets = M[test_x1, test_x2]

    torch.set_rng_state(old_state)

    return M, (train_x1, train_x2, train_targets), (test_x1, test_x2, test_targets)


class BinOpDataset(Dataset):
    def __init__(self, p, op_type='add'):
        self.p = p
        self.op_type = op_type

        self.data = torch.tensor([(x1, x2, self.op(x1, x2)) for x1, x2 in self.op_domain()])
        self.x1 = self.data[:, 0]
        self.x2 = self.data[:, 1]
        self.target = self.data[:, 2]
    
    def op_domain(self):
        if self.op_type == 'div':
            return [(x1, x2) for x1 in range(self.p) for x2 in range(1, self. p)]
        else:
            return [(x1, x2) for x1 in range(self.p) for x2 in range(self. p)]
    
    def op(self, x1: int, x2: int):
        if self.op_type == 'add':
            return (x1 + x2) % self.p
        elif self.op_type == 'max':
            return max(x1, x2)
        elif self.op_type == 'x':
            return x1
        elif self.op_type == 'x2+xy':
            return (x1 ** 2 + x1 * x2) % self.p
        elif self.op_type == 'x3+xy':
            return (x1 ** 3 + x1 * x2) % self.p
        elif self.op_type == 'div':
            for y in range(self.p):
                if (y * x2) % self.p == x1:
                    return y
            assert False
        elif self.op_type == 'zero':
            return 0
        elif self.op_type == 'rand':
            return torch.randint(0, self.p, size=[]).item()
        elif self.op_type == 'tricky':
            return 1 if x1 == 0 or x2 == 0 else 0
    
    def __getitem__(self, i):
        return self.x1[i], self.x2[i], self.target[i]
    
    def __len__(self):
        return self.data.shape[0]


def gen_mod_add(seed, n_train, p):
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    dataset = BinOpDataset(p, 'add')
    
    data_perm = torch.randperm(len(dataset)).tolist()
    train_indices = data_perm[:n_train]
    test_indices = data_perm[n_train:]

    train_inputs1, train_inputs2, train_targets = dataset[train_indices]
    train_inputs = F.one_hot(train_inputs1, p * 2) + F.one_hot(train_inputs2 + p, p * 2)
    train_inputs = train_inputs.to(torch.get_default_dtype())

    test_inputs1, test_inputs2, test_targets = dataset[test_indices]
    test_inputs = F.one_hot(test_inputs1, p * 2) + F.one_hot(test_inputs2 + p, p * 2)
    test_inputs = test_inputs.to(torch.get_default_dtype())

    torch.set_rng_state(old_state)

    return (train_inputs, train_targets), (test_inputs, test_targets)

