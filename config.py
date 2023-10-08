import os
import argparse


root_path = os.path.abspath(os.path.dirname(__file__))
log_path = os.path.join(root_path, 'log/log.log')

# ========================= Data Configs ==========================
data_path = os.path.join(root_path, 'data/phase1_gdata.npz')

# ========================== BERT Configs =============================
torch_model_dir = os.path.join(root_path, 'libs/nezha-cn-base')
save_model_dir = os.path.join(root_path, 'save_models')

pretrained_model = 'auc_0.76194_loss_0.73239_epoch_5_pretrain_model.bin'
trained_model = 'epoch_200_model.bin'
submit_path = os.path.join(root_path, 'data/submit.npy')
dataset_name = 'DGraphFin'


def parser_args():
    parser = argparse.ArgumentParser(description='信也2022图算法 config')

    parser.add_argument('--root_path', type=str, default=root_path)
    parser.add_argument('--log_path', type=str, default=log_path)

    # ========================= Data Configs ==========================
    parser.add_argument('--data_path', type=str, default=data_path)

    # ========================== BERT Configs =============================
    parser.add_argument('--torch_model_dir', type=str, default=torch_model_dir)

    # ========================== Train Configs =============================
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--not_train_batch_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:5')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--save_model_dir', type=str, default=save_model_dir)
    parser.add_argument('--pretrained_model', type=str, default=pretrained_model)
    parser.add_argument('--trained_model', type=str, default=trained_model)
    parser.add_argument('--dataset_name', type=str, default=dataset_name)

    # ========================== Interface Configs =============================
    parser.add_argument('--submit_path', type=str, default=submit_path)

    return parser.parse_args()

