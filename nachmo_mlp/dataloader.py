import os

from torch.utils.data import DataLoader


def prepare_dataloader(dataset, batch_size=256, shuffle=True, num_workers=0):
    # os.environ['NUMEXPR_MAX_THREADS'] = '1'
    print("num_of_workers = ", num_workers, "CPU_avail = ", os.cpu_count())
    loader = DataLoader(
        dataset, drop_last=True, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )  # FIXME why drop_last?

    return loader
