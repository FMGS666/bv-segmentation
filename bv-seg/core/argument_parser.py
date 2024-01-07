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
            '-ptp', 
            '---pretrained-path',
            type = str,
            help = "The path where to load the pretrained weights from",
            required = False,
            default = "./models/pretrained/model_swinvit.pt"
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
            '-lp', 
            '--left-pad',
            type = int,
            help = "left padding for the test volumes",
            required = False,
            default = 62
        )
        self.add_argument(
            '-rp', 
            '--right-pad',
            type = int,
            help = "right padding for the test volumes",
            required = False,
            default = 63
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
            help = "The number of 3D volumes to sampleTifd for each split",
            required = False,
            default = 1
        )
        self.add_argument(
            '-sb', 
            '--subsample',
            help = "Whether to subsample the full slices for creating volumes",
            action = "store_true"
        )
        self.add_argument(
            '-vp', 
            '--volumes-path',
            help = "The path to the 3D volumes NIFTI files",
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
            help = "The weight decay for the optimizer",
            required = False,
            default = 1e-5
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
        self.add_argument(
            '-o', 
            '--overlap',
            type = float,
            help = "The amount of overlap between windows for sliding window inference",
            required = False,
            default = 0.7
        )
        self.add_argument(
            '-wp', 
            '--warmup-period',
            type = int,
            help = "The warmup period",
            required = False,
            default = 50
        )
        self.add_argument(
            '-lpt', 
            '--load-pre-trained',
            help = "Whether to load pre-trained weights",
            action = "store_true"
        )
        self.add_argument(
            '-ddp', 
            '--data-parallel',
            help = "Whether to train using multiple GPUs (single machine)",
            action = "store_true"
        )
        self.add_argument(
            '-s',
            '--skip',
            help = "List of datasets to skip",
            nargs = '+'
        )
        self.add_argument(
            '-f',
            '--feature-size',
            help = "The feature size for the SwinUnetR model",
            type = int,
            default = 48
        )
        self.add_argument(
            '-ptpd',
            '--paths-to-predictors',
            help = "List of paths which to load the predictors from",
            nargs = '+'
        )
        self.add_argument(
            '-wtv', 
            '--write-test-volumes',
            help = "Whether to write test volumes during prediction",
            action = "store_true"
        )
        self.add_argument(
            '-sbn', 
            '---submission-name',
            type = str,
            help = "The name of the submission",
            required = False,
            default = "submission"
        )
        self.add_argument(
            '-sbp', 
            '---submission-path',
            type = str,
            help = "The path to the submission",
            required = False,
            default = "./submissions"
        )