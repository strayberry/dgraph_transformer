import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def collate_fn(batch):
    x = [d['x'] for d in batch]
    y = [d['y'] for d in batch]
    start_edge_timestamp = [d['start_edge_timestamp'] for d in batch]
    end_edge_timestamp = [d['end_edge_timestamp'] for d in batch]
    edge_start_type = [d['edge_start_type'] for d in batch]
    edge_end_type = [d['edge_end_type'] for d in batch]
    back_x = [d['back_x'] for d in batch]
    front_x = [d['front_x'] for d in batch]
    back_x_attention_mask = [d['back_x_attention_mask'] for d in batch]
    front_x_attention_mask = [d['front_x_attention_mask'] for d in batch]

    start_edge_timestamp_max_length = max([len(i) for i in start_edge_timestamp])
    end_edge_timestamp_max_length = max([len(i) for i in end_edge_timestamp])
    edge_start_type_max_length = max([len(i) for i in edge_start_type])
    edge_end_type_max_length = max([len(i) for i in edge_end_type])
    back_x_max_length = max([len(i) for i in back_x])
    front_x_max_length = max([len(i) for i in front_x])

    pad_start_edge_timestamp = [i + [0] * max(start_edge_timestamp_max_length - len(i), 0) for i in
                                start_edge_timestamp]
    pad_end_edge_timestamp = [i + [0] * max(end_edge_timestamp_max_length - len(i), 0) for i in end_edge_timestamp]
    pad_edge_start_type = [i + [0] * max(edge_start_type_max_length - len(i), 0) for i in edge_start_type]
    pad_edge_end_type = [i + [0] * max(edge_end_type_max_length - len(i), 0) for i in edge_end_type]
    pad_back_x = [i + [[0] * 167] * max(back_x_max_length - len(i), 0) for i in back_x]
    pad_front_x = [i + [[0] * 167] * max(front_x_max_length - len(i), 0) for i in front_x]
    pad_back_x_attention_mask = [i + [0] * max(back_x_max_length - len(i), 0) for i in back_x_attention_mask]
    pad_front_x_attention_mask = [i + [0] * max(front_x_max_length - len(i), 0) for i in front_x_attention_mask]
    attention_mask = torch.cat(
        [
            torch.tensor(pad_front_x_attention_mask),
            torch.ones(len(pad_front_x_attention_mask)).unsqueeze(1),
            torch.tensor(pad_back_x_attention_mask)
        ], dim=1)
    return {
        'x': torch.tensor(x).to(torch.float),
        'y': torch.tensor(y).to(torch.long),
        'start_edge_timestamp': torch.tensor(pad_start_edge_timestamp).to(torch.long),
        'end_edge_timestamp': torch.tensor(pad_end_edge_timestamp).to(torch.long),
        'edge_start_type': torch.tensor(pad_edge_start_type).to(torch.long),
        'edge_end_type': torch.tensor(pad_edge_end_type).to(torch.long),
        'back_x': torch.tensor(pad_back_x).to(torch.float),
        'front_x': torch.tensor(pad_front_x).to(torch.float),
        'attention_mask': attention_mask
    }
