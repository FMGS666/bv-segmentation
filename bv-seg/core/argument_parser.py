f"""

file: {__file__}

Contents: 
    *  BVSegArgumentParser

Argument parser for our `bv-seg` module


The available options are the following:
  -h, --help            show this help message and exit

  -trdp TRAIN_DATA_PATH, --train-data-path TRAIN_DATA_PATH
                        Path to the raw train data

  -tsdp TEST_DATA_PATH, --test-data-path TEST_DATA_PATH
                        Path to the raw train data

  -k K, --K K           K for the K-Fold Cross Validation

  -r RANDOM_STATE, --random-state RANDOM_STATE
                        Random state for the K-Fold Cross Validation

  -S, --shuffle         Whether to shuffle observations or not for the K-Fold Cross Validation

  -tbs TRAIN_BATCH_SIZE, --train-batch-size TRAIN_BATCH_SIZE
                        The batch size for the training data loaders

  -vbs VALIDATION_BATCH_SIZE, --validation-batch-size VALIDATION_BATCH_SIZE
                        The batch size for the validation data loaders

  -o OPTIMIZER_ID, --optimizer-id OPTIMIZER_ID
                        The id of the optimizer to be used

  -s SCHEDULER_ID, --scheduler-id SCHEDULER_ID
                        The id of the scheduler to be used

  -e EPOCHS, --epochs EPOCHS
                        The number of epochs which to train the model over

  -p PATIENCE, --patience PATIENCE
                        The patience for early stopping

  -lr INITIAL_LEARNING_RATE, --initial-learning-rate INITIAL_LEARNING_RATE
                        The learning rate which to start the tuning with

  -g GAMMA, --gamma GAMMA
                        The gamma for the ExponentialLR scheduler

  -a ALPHA, --alpha ALPHA
                        The alpha for the RMSProp optimize

  -P POWER, --power POWER
                        The power for the PolynomialLR scheduler

  -st, --sched-step-after-train
                        Whether to shuffle observations or not for the K-Fold Cross Validation

  -ri, --relative-improvement
                        Whether to use relative improvement as stopping criterion

  -n MODEL_NAME, --model-name MODEL_NAME
                        The name for the model

  -dp DUMP_PATH, --dump-path DUMP_PATH
                        The path where to save the models

  -lp LOG_PATH, --log-path LOG_PATH
                        The path where to save logs during training

  -smp SPLITS_METADATA_PATH, --splits-metadata-path SPLITS_METADATA_PATH
                        The path where to save splits metadata

  -c CONTEXT_LENGTH, --context-length CONTEXT_LENGTH
                        The context length for sampling 3D volumes

  -ns N_SAMPLES, --n-samples N_SAMPLES
                        The number of 3D volumes to sample for each split
                        
  -wv, --write-volumes  Whether to write to disk new 3D volumes sampled from each split

  -vp VOLUMES_PATH, --volumes-path VOLUMES_PATH
                        The path to the 3D volumes Tif files

  -dm, --dump-metadata  Whether to dump volumes metadata

  -t, --train           Whether to perform training

"""
import argparse

class BVSegArgumentParser(argparse.ArgumentParser):
    def __init__(
            self,
            supported_commands: list[str],
            prog: str = "This software can be used for training models for the Blood Vessel Segmentation Kaggle competition",
            description: str = "We'll make a better description later, sorry folks",
            epilog: str = "If you're reading this, it's far too late dawg"
        ) -> None:
        super(BVSegArgumentParser, self).__init__(
            prog = prog,
            description = description,
            epilog = epilog
        )
        self.supported_commands = supported_commands
        self.add_argument(
            'command',
            choices = self.supported_commands,
            help = f"The command to be run. The supported commands are: {self.supported_commands}",
        )
        self.add_argument(
            '-mbp', 
            '---metadata-base-path',
            type = str,
            help = "The path where to save splits metadata",
            required = False,
            default = "./data/splits_metadata/"
        )

        self.add_argument(
            '-trdp', 
            '--train-data-path',
            type = str,
            help = "Path to the raw train data",
            required = False,
            default = "./data/train"
        )
        self.add_argument(
            '-tsdp', 
            '--test-data-path',
            type = str,
            help = "Path to the raw train data",
            required = False,
            default = "./data/test"
        )
        self.add_argument(
            '-k', 
            '--K',
            type = int,
            help = "K for the K-Fold Cross Validation",
            required = False,
            default = 3
        )
        self.add_argument(
            '-r', 
            '--random-state',
            type = int,
            help = "Random state for the K-Fold Cross Validation",
            required = False,
            default = None
        )
        self.add_argument(
            '-S', 
            '--shuffle',
            help = "Whether to shuffle observations or not for the K-Fold Cross Validation",
            action = "store_true"
        )
        self.add_argument(
            '-c', 
            '--context-length',
            type = int,
            help = "The context length for sampling 3D volumes",
            required = False,
            default = 50
        ) 
        self.add_argument(
            '-ns', 
            '--n-samples',
            type = int,
            help = "The number of 3D volumes to sample for each split",
            required = False,
            default = 1
        )
        self.add_argument(
            '-sb', 
            '--subsample',
            help = "Whether to take full slices for creating volumes",
            action = "store_true"
        ) 
        self.add_argument(
            '-vp', 
            '--volumes-path',
            help = "The path to the 3D volumes Tif files",
            type = str,
            required = False,
            default = "./data/splits_sampled_volumes"
        )
        self.add_argument(
            '-tbs', 
            '--train-batch-size',
            type = int,
            help = "The batch size for the training data loaders",
            required = False,
            default = 1
        )
        self.add_argument(
            '-vbs', 
            '--validation-batch-size',
            type = int,
            help = "The batch size for the validation data loaders",
            required = False,
            default = 1
        )
        self.add_argument(
            '-e', 
            '--epochs',
            type = int,
            help = "The number of epochs which to train the model over",
            required = False,
            default = 800
        )
        self.add_argument(
            '-p', 
            '--patience',
            type = int,
            help = "The patience for early stopping",
            required = False,
            default = 5
        )
        self.add_argument(
            '-lr', 
            '--initial-learning-rate',
            type = float,
            help = "The learning rate which to start the tuning with",
            required = False,
            default = 8e-4
        )
        self.add_argument(
            '-wd', 
            '--weight-decay',
            type = float,
            help = "The weight decay for the Adam(W) optimizers",
            required = False,
            default = 1e-5
        )
        self.add_argument(
            '-st', 
            '--sched-step-after-train',
            help = "Whether to shuffle observations or not for the K-Fold Cross Validation",
            action = "store_true"
        )
        self.add_argument(
            '-ri', 
            '--relative-improvement',
            help = "Whether to use relative improvement as stopping criterion",
            action = "store_true"
        )
        self.add_argument(
            '-n', 
            '--model-name',
            type = str,
            help = "The name for the model",
            required = False,
            default = "Swin UnetR"
        ) 
        self.add_argument(
            '-dp', 
            '--dump-path',
            type = str,
            help = "The path where to save the models",
            required = False,
            default = "./models"
        ) 
        self.add_argument(
            '-lp', 
            '--log-path',
            type = str,
            help = "The path where to save logs during training",
            required = False,
            default = "./logs"
        )
        self.add_argument(
            '-ps', 
            '--patch-size',
            type = int,
            help = "The size of the patches",
            required = False,
            default = 128
        )
        
