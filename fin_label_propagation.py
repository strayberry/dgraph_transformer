import numpy as np

from logger import logger

from config import parser_args
from utils.graph_dataset import GraphDataset


if __name__ == '__main__':
    args = parser_args()
    logger.info(args)

    dataset = GraphDataset(args)
    #dataset.split_data()
    adjacency_matrix = dataset.build_sparse_adjacency_matrix(dataset.edge_index, len(dataset.x))
    print(adjacency_matrix)
    enhanced_adjacency_matrix = dataset.add_edges_with_knn(adjacency_matrix, k=5)
    print(enhanced_adjacency_matrix)
    dataset.label_propagation_async(enhanced_adjacency_matrix)


    np.savez(
        'labelled_dataset.npz', 
        x=dataset.data_x, 
        y=dataset.y, 
        edge_index=dataset.edge_index,
        edge_type=dataset.edge_type,
        edge_timestamp=dataset.edge_timestamp,
        train_mask=dataset.train_new_mask,
        test_mask=dataset.test_new_mask
        )
    print("train_new_mask.shape:", dataset.train_new_mask.shape)
    print("test_new_mask.shape:", dataset.test_new_mask.shape)
