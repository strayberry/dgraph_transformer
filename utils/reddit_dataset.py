import os
import torch
from utils import utils as u
#import utils as u
from datetime import datetime
from torch.utils.data import Dataset


class RedditDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.reddit_args = args.reddit_args
        self.folder = self.reddit_args['folder']
        self.edges, self.feats, self.num_nodes, self.max_time = self.load_edges_and_features()

    def __len__(self):
        return len(self.edges['idx'])

    def __getitem__(self, idx):
        edge_idx = self.edges['idx'][idx]
        edge_vals = self.edges['vals'][idx]

        source, target, time, label = edge_idx

        x = self.feats[source].tolist()
        y = label.tolist()
        start_edge_timestamp = [time.tolist()]
        end_edge_timestamp = [time.tolist()]
        edge_start_type = [label.tolist()]
        edge_end_type = [label.tolist()]
        back_x = self.feats[target].tolist()
        front_x = x
        back_x_attention_mask = [1] * len(back_x)
        front_x_attention_mask = [1] * len(front_x)

        return {
            'x': x,
            'y': y,
            'start_edge_timestamp': start_edge_timestamp,
            'end_edge_timestamp': end_edge_timestamp,
            'edge_start_type': edge_start_type,
            'edge_end_type': edge_end_type,
            'back_x': back_x,
            'front_x': front_x,
            'back_x_attention_mask': back_x_attention_mask,
            'front_x_attention_mask': front_x_attention_mask
        }

    def load_edges_and_features(self):
        folder = self.reddit_args['folder']

        # Load nodes
        cols = u.Namespace({'id': 0, 'feats': 1})
        file = self.reddit_args['feats_file']
        file = os.path.join(folder, file)
        with open(file) as file:
            file = file.read().splitlines()

        ids_str_to_int = {}
        id_counter = 0
        feats = []

        for line in file:
            line = line.split(',')
            # Node id
            nd_id = line[0]
            if nd_id not in ids_str_to_int.keys():
                ids_str_to_int[nd_id] = id_counter
                id_counter += 1
                nd_feats = [float(r) for r in line[1:]]
                feats.append(nd_feats)
            else:
                print('duplicate id', nd_id)
                raise Exception('duplicate_id')

        feats = torch.tensor(feats, dtype=torch.float)
        num_nodes = feats.size(0)

        self.ids_str_to_int = ids_str_to_int
        self.num_nodes = num_nodes

        edges = []
        not_found = 0
        # Load edges in title
        edges_tmp, not_found_tmp = self.load_edges_from_file(self.reddit_args['title_edges_file'], folder)
        edges.extend(edges_tmp)
        not_found += not_found_tmp

        # Load edges in bodies
        edges_tmp, not_found_tmp = self.load_edges_from_file(self.reddit_args['body_edges_file'], folder)
        edges.extend(edges_tmp)
        not_found += not_found_tmp

        # min time should be 0 and time aggregation
        edges = torch.LongTensor(edges)
        edges[:, 2] = u.aggregate_by_time(edges[:, 2], self.reddit_args['aggr_time'])
        max_time = edges[:, 2].max()

        # separate classes
        sp_indices = edges[:, :3].t()
        sp_values = edges[:, 3]

        pos_mask = sp_values == 1
        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:, neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices,
                                            neg_sp_values,
                                            torch.Size([self.num_nodes,
                                                        self.num_nodes,
                                                        max_time + 1])).coalesce()

        pos_sp_indices = sp_indices[:, pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices,
                                            pos_sp_values,
                                            torch.Size([self.num_nodes,
                                                        self.num_nodes,
                                                        max_time + 1])).coalesce()

        # scale positive class to separate after adding
        pos_sp_edges *= 1000

        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        # separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals % 1000
        pos_vals = vals // 1000
        vals = pos_vals - neg_vals

        # creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0), dtype=torch.long)
        new_vals[vals > 0] = 1
        new_vals[vals <= 0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(), new_vals.view(-1, 1)], dim=1)

        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_classes = 2
        self.feats_per_node = feats.size(1)
        self.nodes_feats = feats
        self.max_time = max_time
        self.min_time = 0

        return self.edges, feats, num_nodes, max_time

    def load_edges_from_file(self, edges_file, folder):
        edges = []
        not_found = 0

        file = os.path.join(folder, edges_file)
        with open(file) as file:
            file = file.read().splitlines()

        cols = u.Namespace({'source': 0, 'target': 1, 'time': 3, 'label': 4})
        base_time = datetime.strptime("19800101", '%Y%m%d')

        for line in file[1:]:
            fields = line.split('\t')
            sr = fields[cols.source]
            tg = fields[cols.target]

            if sr in self.ids_str_to_int.keys() and tg in self.ids_str_to_int.keys():
                sr = self.ids_str_to_int[sr]
                tg = self.ids_str_to_int[tg]

                time = fields[cols.time].split(' ')[0]
                time = datetime.strptime(time, '%Y-%m-%d')
                time = (time - base_time).days

                label = int(fields[cols.label])
                edges.append([sr, tg, time, label])
                edges.append([tg, sr, time, label])
            else:
                not_found += 1

        return edges, not_found


# Usage example:
if __name__ == "__main__":
    class Args:
        reddit_args = {
            'folder': '/home/ubuntu/2022_finvcup_baseline/data/reddit/',
            'feats_file': 'web-redditEmbeddings-subreddits.csv',
            'title_edges_file': 'soc-redditHyperlinks-title.tsv',
            'body_edges_file': 'soc-redditHyperlinks-body.tsv',
            'aggr_time': 1  # Set the appropriate aggregation time
        }

    args = Args()
    reddit_dataset = RedditDataset(args)
    example_idx = 0  # Replace with the desired example index
    example_data = reddit_dataset[example_idx]
    print(example_data)  # Print the example data for demonstration
