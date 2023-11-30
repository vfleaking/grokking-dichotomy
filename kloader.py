import torch
import torch.utils.data

import time


class KDataLoader:
    TOTAL_LOAD_TIME = 0
    
    def _iter(self, num):
        raise NotImplementedError()
    

    def iter(self, num=None):
        if num is not None:
            while num > len(self):
                yield from self.iter()
                num -= len(self)
        
        last = time.time()
        for ret in self._iter(num):
            KDataLoader.TOTAL_LOAD_TIME += time.time() - last
            yield ret
            last = time.time()


    def enum(self, num=None):
        yield from enumerate(self.iter(num))
    
    
    def __len__(self):
        raise NotImplementedError()
    
    
    def dataset_size(self):
        raise NotImplementedError()


class KDataLoaderTorch(KDataLoader):
    loader: torch.utils.data.DataLoader
    
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        if hasattr(self.loader.sampler, 'set_epoch'):
            self.sampler = self.loader.sampler
            self.cnt = 0
        else:
            self.sampler = None
    
    def _iter(self, num):
        if self.sampler is not None:
            self.sampler.set_epoch(self.cnt)
            self.cnt += 1
        if num is None:
            for idx, pt in enumerate(self.loader):
                yield tuple(p.to(self.device) for p in pt)
        else:
            it = iter(self.loader)
            for idx, pt in enumerate(self.loader):
                if idx >= num:
                    del it
                    break
                yield tuple(p.to(self.device) for p in pt)

    def __len__(self):
        return len(self.loader)
    
    def dataset_size(self):
        return len(self.loader.dataset)


class KTensorDataLoader(KDataLoader):
    
    def __init__(self, data, batch_size, shuffle=True, drop_last=True, device=None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
    
    def move_to_device(self, x):
        if self.device is not None:
            x = x.to(self.device)
        return x
    
    def _iter(self, num):
        if self.shuffle:
            perm = torch.randperm(self.dataset_size())
            
        n = self.dataset_size() // self.batch_size
        for idx in range(n):
            if num is not None and idx >= num:
                break
            if self.shuffle:
                pm = perm[idx * self.batch_size : (idx + 1) * self.batch_size]
                yield tuple(self.move_to_device(p[pm]) for p in self.data)
            else:
                yield tuple(self.move_to_device(p[idx * self.batch_size : (idx + 1) * self.batch_size]) for p in self.data)
        
        if num is None or n < num:
            if not self.drop_last and n * self.batch_size < self.dataset_size():
                if self.shuffle:
                    pm = perm[n * self.batch_size:]
                    yield tuple(self.move_to_device(p[pm]) for p in self.data)
                else:
                    yield tuple(self.move_to_device(p[n * self.batch_size:]) for p in self.data)
    
    def __len__(self):
        if self.drop_last:
            return self.dataset_size() // self.batch_size
        else:
            return (self.dataset_size() + self.batch_size - 1) // self.batch_size
    
    def dataset_size(self):
        return self.data[0].shape[0]