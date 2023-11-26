import argparse

class BVSegArgumentParser(argparse.ArgumentParser):
    def __init__(
            self,
            prog: str = "This software can be used for training models for the Blood Vessel Segmentation Kaggle competition",
            description: str = "We'll make a better description later, sorry folks",
            epilog: str = "If you're reading this, it's far too late dawg"
        ) -> None:
        super(BVSegArgumentParser, self).__init__(
            prog = prog,
            description = description,
            epilog = epilog
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
            '-o', 
            '--optimizer-id',
            type = str,
            help = "The id of the optimizer to be used",
            required = False,
            default = "AdamW"
        )
        self.add_argument(
            '-s', 
            '--scheduler-id',
            type = str,
            help = "The id of the scheduler to be used",
            required = False,
            default = None
        )
        self.add_argument(
            '-e', 
            '--epochs',
            type = int,
            help = "The number of epochs which to train the model over",
            required = False,
            default = 100
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
            default = 1e-4
        )
        self.add_argument(
            '-g', 
            '--gamma',
            type = float,
            help = "The gamma for the ExponentialLR scheduler",
            required = False,
            default = 9999e-4
        )
        self.add_argument(
            '-a', 
            '--alpha',
            type = float,
            help = "The alpha for the RMSProp optimize",
            required = False,
            default = 9e-1
        )
        self.add_argument(
            '-P', 
            '--power',
            type = float,
            help = "The power for the PolynomialLR scheduler",
            required = False,
            default = 2.0
        )
        self.add_argument(
            '-st', 
            '--sched-step-after-train',
            help = "Whether to shuffle observations or not for the K-Fold Cross Validation",
            action = "store_true"
        )
        self.add_argument(
            '-stk', 
            '--stacked',
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
            default = "Unet"
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
            '-smp', 
            '--splits-metadata-path',
            type = str,
            help = "The path where to save splits metadata",
            required = False,
            default = "./bv-seg/splits_metadata"
        ) 
        
