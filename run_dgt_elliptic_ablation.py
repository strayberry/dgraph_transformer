from logger import logger
from train_elliptic_dgt import train_all
from config import parser_args
from utils.elliptic_dataset import EllipticTemporalDataset


configs = [
    ("Without time features", {"use_time_features": False}),
    ("Without time ordering", {"use_time_ordering": False}),
    ("Without BiLSTM", {"use_bilstm": False}),
    ("Without pretraining", {"use_pretraining": False}),
]

if __name__ == '__main__':
    args = parser_args()
    args.elliptic_args = {'folder': '/home/ubuntu/2022_finvcup_baseline/data/elliptic_temporal','tar_file': 'elliptic_bitcoin_dataset_cont.tar.gz','classes_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv','times_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv','edges_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv','feats_file': 'elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv'}
    args.dataset = EllipticTemporalDataset(args)
    args.dataset_name = 'Elliptic'
    logger.info(args)

    for experiment_name, conf in configs:
        print(f"Running experiment: {experiment_name}")
        
        # Update the args values based on the current configuration
        for key, value in conf.items():
            setattr(args, key, value)

        train_all(args)
