from logger import logger
from train_fin_dgt import train
from config import parser_args

configs = [
    ("Without time features", {"use_time_features": False}),
    ("Without time ordering", {"use_time_ordering": False}),
    ("Without BiLSTM", {"use_bilstm": False}),
    ("Without pretraining", {"use_pretraining": False}),
]

if __name__ == '__main__':
    args = parser_args()
    logger.info(args)

    for experiment_name, conf in configs:
        print(f"Running experiment: {experiment_name}")
        
        # Update the args values based on the current configuration
        for key, value in conf.items():
            setattr(args, key, value)

        train(args)
