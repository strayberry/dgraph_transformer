import os
import argparse


root_path = os.path.abspath(os.path.dirname(__file__))
log_path = os.path.join(root_path, 'log/log.log')

# ========================= Data Configs ==========================
data_path = os.path.join(root_path, 'data/phase1_gdata.npz')

# ========================== BERT Configs =============================
torch_model_dir = os.path.join(root_path, 'libs/nezha-cn-base')
save_model_dir = os.path.join(root_path, 'save_models')

pretrained_model = 'auc_0.75025_loss_1.05465_epoch_5_pretrain_model.bin'
trained_model = 'epoch_10_model.bin'
submit_path = os.path.join(root_path, 'data/submit.npy')
dataset_name = 'DGraphFin'


def parser_args():
    parser = argparse.ArgumentParser(description='DGT config')

    parser.add_argument('--root_path', type=str, default=root_path)
    parser.add_argument('--log_path', type=str, default=log_path)

    # ========================= Data Configs ==========================
    parser.add_argument('--data_path', type=str, default=data_path)

    # ========================== BERT Configs =============================
    parser.add_argument('--torch_model_dir', type=str, default=torch_model_dir)

    # ========================== Train Configs =============================
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--not_train_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--num_works', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:8')
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--save_model_dir', type=str, default=save_model_dir)
    parser.add_argument('--pretrained_model', type=str, default=pretrained_model)
    parser.add_argument('--trained_model', type=str, default=trained_model)
    parser.add_argument('--dataset_name', type=str, default=dataset_name)

    # ========================== Interface Configs =============================
    parser.add_argument('--submit_path', type=str, default=submit_path)

    # ========================== Ablation =============================
    parser.add_argument('--use_time_features', default=True, type=bool, help='Whether to use time features')
    parser.add_argument('--use_time_ordering', default=True, type=bool, help='Whether to use time ordering')
    parser.add_argument('--use_bilstm', default=True, type=bool, help='Whether to use BiLSTM layer')
    parser.add_argument('--use_pretraining', default=True, type=bool, help='Whether to use pre-trained weights')

    return parser.parse_args()

