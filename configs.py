import argparse
import copy
import datetime
import json
import os


class ConfigBase(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, "hash"):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Create a configuration object from command line arguments."""
        parents = [
            cls.ddp_parser(),
            cls.data_parser(),  # task-agnostic
            cls.model_parser(),  # task-agnostic
            cls.train_parser(),  # task-agnostic
            cls.logging_parser(),  # task-agnostic
            cls.task_specific_parser(),
        ]

        parser = argparse.ArgumentParser(
            add_help=True, parents=parents, fromfile_prefix_chars="@"
        )
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(
            namespace=config
        )  # sets parsed arguments as attributes of namespace

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, "r") as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, "configs.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs["task"] = self.task
        attrs["model_name"] = self.model_name
        attrs["checkpoint_dir"] = self.checkpoint_dir

        with open(path, "w") as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self.backbone_type

    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root, self.data, self.task, self.model_name, self.hash
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def ddp_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Data Distributed Training", add_help=False)
        parser.add_argument("--gpus", type=int, nargs="+", default=None, help="")
        parser.add_argument(
            "--server",
            type=str,
            choices=("main", "workstation1", "workstation2", "workstation3"),
        )
        parser.add_argument("--num-nodes", type=int, default=1, help="")
        parser.add_argument("--node-rank", type=int, default=0, help="")
        parser.add_argument(
            "--dist-url", type=str, default="tcp://127.0.0.1:3500", help=""
        )
        parser.add_argument("--dist-backend", type=str, default="nccl", help="")

        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument("--root", type=str, default="./datasets")
        parser.add_argument(
            "--data",
            type=str,
            default="cifar10",
            choices=("cifar10", "cifar100", "tiny", "svhn", "imagenet"),
        )
        parser.add_argument("--mismatch-ratio", type=float, default=0.30)
        parser.add_argument("--n-label-per-class", type=int, default=400)
        parser.add_argument("--n-valid-per-class", type=int, default=None, help="10%")
        parser.add_argument(
            "--input-size", type=int, default=32, choices=(32, 64, 96, 224)
        )
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument(
            "--augmentation",
            type=str,
            default="torchvision",
            choices=("torchvision", "albumentations"),
            help="Package used for augmentation.",
        )
        parser.add_argument(
            "--convert-filename",
            type=str,
            default="datasets/imagenet32_filter.csv",
            help="class matching dictionary only for imagenet32 dataset",
        )
        parser.add_argument(
            "--n-bins", type=int, default=15, help="Expected calibration error, n-bins"
        )

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("CNN Backbone", add_help=False)
        parser.add_argument(
            "--backbone-type",
            type=str,
            default="wide28_2",
            choices=(
                "wide28_2",
                "wide28_10",
                "densenet121",
                "vgg16_bn",
                "inceptionv4",
                "resnet50",
                "vit"
            ),
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Path to checkpoint file to resume training from.",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="L2 Normalize."
        )

        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument(
            "--iterations", type=int, default=500000, help="Number of training epochs."
        )
        parser.add_argument(
            "--warm-up", type=int, default=200000, help="Number of training epochs."
        )
        parser.add_argument(
            "--batch-size", type=int, default=100, help="Mini-batch size."
        )
        parser.add_argument(
            "--num-workers", type=int, default=8, help="Number of CPU threads."
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=(
                "sgd",
                "adam",
            ),
            help="Optimization algorithm.",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=3e-3,
            help="Base learning rate to start from.",
        )
        parser.add_argument(
            "--mixed-precision", action="store_true", help="Use float16 precision."
        )
        parser.add_argument(
            "--milestones",
            action="store",
            type=int,
            nargs="*",
            default=[400000],
            help="learning rate decay milestones",
        )
        parser.add_argument(
            "--gamma", type=float, default=0.2, help="learning rate decay gamma"
        )
        parser.add_argument(
            "--weight-decay", type=float, default=0, help="l2 weight decay"
        )

        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument(
            "--checkpoint-root",
            type=str,
            default="./checkpoints/",
            help="Top-level directory of checkpoints.",
        )
        parser.add_argument(
            "--save-every",
            type=int,
            default=5000,
            help="Save model checkpoint every `save_every` epochs.",
        )
        parser.add_argument(
            "--enable-wandb", action="store_true", help="Use Weights & Biases plugin."
        )
        parser.add_argument("--wandb-proj-v", type=str, default="")
        parser.add_argument(
            "--enable-plot",
            action="store_true",
            help="Plotting unlabeled and testing dataset - TSNE.",
        )

        return parser


class SLConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SLConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )

        return parser

    @property
    def task(self) -> str:
        return "SL"


class FIXMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(FIXMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--tau", type=float, default=0.95)
        parser.add_argument("--consis-coef", type=float, default=1)
        parser.add_argument("--start-fix", type=int, default=5)

        return parser

    @property
    def task(self) -> str:
        return "FIXMATCH"


class AdelloConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(AdelloConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--start-fix", type=int, default=5)

        return parser

    @property
    def task(self) -> str:
        return "Adello"


class CaliMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(CaliMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--tau", type=float, default=0.95)
        parser.add_argument("--consis-coef", type=float, default=1)
        parser.add_argument(
            "--train-n-bins",
            type=int,
            default=30,
            help="Expected calibration error, n-bins in AcatS.",
        )
        parser.add_argument("--swa-on", action="store_true", help="swa optimizer on.")
        parser.add_argument("--swa-start-iter", default=100000, type=int)

        return parser

    @property
    def task(self) -> str:
        return "CaliMATCH"


class ProposedConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(ProposedConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )

        parser.add_argument("--tau", type=float, default=0.95)
        parser.add_argument("--pi", type=float, default=0.5)

        parser.add_argument(
            "--train-n-bins",
            type=int,
            default=30,
            help="Expected calibration error, n-bins in AcatS.",
        )
        parser.add_argument("--start-fix", type=int, default=5)

        parser.add_argument("--lambda-cali", type=float, default=1)
        parser.add_argument("--lambda-ova-soft", type=float, default=5e-1)
        parser.add_argument("--lambda-ova-cali", type=float, default=5e-1)
        parser.add_argument("--lambda-ova", type=float, default=5e-1)
        parser.add_argument("--lambda-fix", type=float, default=1)

        return parser

    @property
    def task(self) -> str:
        return "Proposed"


class ProposedPlusConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(ProposedPlusConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )

        parser.add_argument("--tau", type=float, default=0.95)

        parser.add_argument(
            "--train-n-bins",
            type=int,
            default=30,
            help="Expected calibration error, n-bins in AcatS.",
        )
        parser.add_argument("--start-fix", type=int, default=5)

        parser.add_argument("--lambda-cali", type=float, default=1)
        parser.add_argument("--lambda-ova-soft", type=float, default=5e-1)
        parser.add_argument("--lambda-ova-cali", type=float, default=5e-1)
        parser.add_argument("--lambda-ova", type=float, default=5e-1)
        parser.add_argument("--lambda-fix", type=float, default=1)

        return parser

    @property
    def task(self) -> str:
        return "ProposedPlus"


class Ablation1Config(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(Ablation1Config, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )

        parser.add_argument("--tau", type=float, default=0.95)
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument(
            "--train-n-bins",
            type=int,
            default=30,
            help="Expected calibration error, n-bins in AcatS.",
        )

        parser.add_argument("--lambda-ova-soft", type=float, default=5e-1)
        parser.add_argument("--lambda-ova-cali", type=float, default=5e-1)
        parser.add_argument("--lambda-ova", type=float, default=5e-1)
        parser.add_argument("--lambda-fix", type=float, default=1)

        return parser

    @property
    def task(self) -> str:
        return "Ablation1"


class Ablation2Config(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(Ablation2Config, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )

        parser.add_argument("--tau", type=float, default=0.95)
        parser.add_argument(
            "--train-n-bins",
            type=int,
            default=30,
            help="Expected calibration error, n-bins in AcatS.",
        )
        parser.add_argument("--start-fix", type=int, default=5)

        parser.add_argument("--lambda-cali", type=float, default=1)
        parser.add_argument("--lambda-ova-soft", type=float, default=5e-1)
        parser.add_argument("--lambda-ova", type=float, default=5e-1)
        parser.add_argument("--lambda-fix", type=float, default=1)

        return parser

    @property
    def task(self) -> str:
        return "Ablation2"


class Ablation3Config(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(Ablation3Config, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )

        parser.add_argument("--tau", type=float, default=0.95)
        parser.add_argument("--start-fix", type=int, default=5)

        parser.add_argument("--lambda-ova-soft", type=float, default=5e-1)
        parser.add_argument("--lambda-ova", type=float, default=5e-1)
        parser.add_argument("--lambda-fix", type=float, default=1)

        return parser

    @property
    def task(self) -> str:
        return "Ablation3"


class Ablation4Config(Ablation1Config):
    def __init__(self, args=None, **kwargs):
        super(Ablation4Config, self).__init__(args, **kwargs)

    @property
    def task(self) -> str:
        return "Ablation4"


class Ablation5Config(Ablation2Config):
    def __init__(self, args=None, **kwargs):
        super(Ablation5Config, self).__init__(args, **kwargs)

    @property
    def task(self) -> str:
        return "Ablation5"


class IOMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(IOMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--q-cutoff", type=float, default=0.50)
        parser.add_argument("--lambda-open", type=float, default=1)
        parser.add_argument("--dist-da-len", type=int, default=128)

        return parser

    @property
    def task(self) -> str:
        return "IOMATCH"


class OPENMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(OPENMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--pi", type=float, default=0.5)
        parser.add_argument("--lambda-em", type=float, default=0.1)
        parser.add_argument(
            "--lambda-socr",
            type=float,
            default=0.5,
            help="SOCR enhances the smoothness of the outlier detector over data augmentation",
        )
        parser.add_argument("--start-fix", type=int, default=5)

        return parser

    @property
    def task(self) -> str:
        return "OPENMATCH"


class SCOMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SCOMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument("--threshold", type=float, default=0.95)
        parser.add_argument("--ood-threshold", type=float, default=0.95)
        parser.add_argument("--Km", type=int, default=1)
        parser.add_argument("--T", type=float, default=1)

        return parser

    @property
    def task(self) -> str:
        return "SCOMATCH"


class SSBConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SSBConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument("--lambda-ova-u", type=float, default=1)
        parser.add_argument("--ova-unlabeled-threshold", type=float, default=0.01)
        parser.add_argument("--lambda-x", type=float, default=1)
        parser.add_argument("--lambda-ova", type=float, default=1)
        parser.add_argument("--lambda-oem", type=float, default=0.1)
        parser.add_argument("--lambda-socr", type=float, default=0.5)
        parser.add_argument("--lambda-u", type=float, default=1)
        parser.add_argument("--T", type=float, default=1)
        parser.add_argument("--threshold", type=float, default=0.95)

        return parser

    @property
    def task(self) -> str:
        return "SSB"


class ACRConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(ACRConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument("--threshold", type=float, default=0.95)
        parser.add_argument("--T", type=float, default=1)
        parser.add_argument(
            "--tau1", default=2, type=float, help="tau for head1 consistency"
        )
        parser.add_argument(
            "--tau12", default=2, type=float, help="tau for head2 consistency"
        )
        parser.add_argument(
            "--tau2", default=2, type=float, help="tau for head2 balanced CE loss"
        )
        parser.add_argument(
            "--ema-u",
            default=0.9,
            type=float,
            help="ema ratio for estimating distribution of the unlabeled data",
        )
        parser.add_argument(
            "--est-epoch",
            default=5,
            type=int,
            help="the start step to estimate the distribution",
        )

        return parser

    @property
    def task(self) -> str:
        return "ACR"


class OPENMATCHMixupConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(OPENMATCHMixupConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--pi", type=float, default=0.5)
        parser.add_argument("--lambda-em", type=float, default=0.1)
        parser.add_argument(
            "--lambda-socr",
            type=float,
            default=0.5,
            help="SOCR enhances the smoothness of the outlier detector over data augmentation",
        )
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument("--alpha", type=float, default=0.2)

        return parser

    @property
    def task(self) -> str:
        return "OPENMATCH+Mixup"


class OPENMATCHSmoothingConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(OPENMATCHSmoothingConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--pi", type=float, default=0.5)
        parser.add_argument("--lambda-em", type=float, default=0.1)
        parser.add_argument(
            "--lambda-socr",
            type=float,
            default=0.5,
            help="SOCR enhances the smoothness of the outlier detector over data augmentation",
        )
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument(
            "--alpha", type=float, default=0.01, choices=(0.01, 0.05, 0.005)
        )

        return parser

    @property
    def task(self) -> str:
        return "OPENMATCH+Smoothing"


class OPENMATCHMMCEConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(OPENMATCHMMCEConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--pi", type=float, default=0.5)
        parser.add_argument("--lambda-em", type=float, default=0.1)
        parser.add_argument(
            "--lambda-socr",
            type=float,
            default=0.5,
            help="SOCR enhances the smoothness of the outlier detector over data augmentation",
        )
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument("--lambda-mmce", type=float, default=9)

        return parser

    @property
    def task(self) -> str:
        return "OPENMATCH+MMCE"


class OPENMATCHRankMixupConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(OPENMATCHRankMixupConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--pi", type=float, default=0.5)
        parser.add_argument("--lambda-em", type=float, default=0.1)
        parser.add_argument(
            "--lambda-socr",
            type=float,
            default=0.5,
            help="SOCR enhances the smoothness of the outlier detector over data augmentation",
        )
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.2,
            help="For mixup, we fix α = 1, which results in interpolations λ uniformly distributed between zero and one.",
        )
        parser.add_argument("--num-mixup", type=int, default=1, help="")
        parser.add_argument("--loss-coef", type=float, default=0.5, help="")

        return parser

    @property
    def task(self) -> str:
        return "OPENMATCH+RankMixup"


class OPENMATCHMbLSConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(OPENMATCHMbLSConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--pi", type=float, default=0.5)
        parser.add_argument("--lambda-em", type=float, default=0.1)
        parser.add_argument(
            "--lambda-socr",
            type=float,
            default=0.5,
            help="SOCR enhances the smoothness of the outlier detector over data augmentation",
        )
        parser.add_argument("--start-fix", type=int, default=5)
        parser.add_argument("--margin", type=float, default=10, help="")
        parser.add_argument("--loss-coef", type=float, default=0.1, help="")

        return parser

    @property
    def task(self) -> str:
        return "OPENMATCH+MbLS"


class PseudoLabelConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(PseudoLabelConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--consis-coef", type=float, default=1)
        parser.add_argument("--threshold", type=float, default=0.95)

        return parser

    @property
    def task(self) -> str:
        return "PseudoLabel"


class VATConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(VATConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--consis-coef", type=float, default=0.3)
        parser.add_argument("--xi", type=float, default=1e-6)
        parser.add_argument("--eps", type=float, default=6)

        return parser

    @property
    def task(self) -> str:
        return "VAT"


class MixMatchConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(MixMatchConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--consis-coef", type=float, default=100)
        parser.add_argument("--alpha", type=float, default=0.75)
        parser.add_argument("--T", type=float, default=0.5)
        parser.add_argument("--K", type=int, default=2)

        return parser

    @property
    def task(self) -> str:
        return "MixMatch"


class TestingConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(TestingConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--checkpoint-hash", type=str, default="2023-01-12_03-30-35", help=""
        )
        parser.add_argument("--for-what", type=str, default="Proposed", required=True)
        parser.add_argument("--safe-student-T", type=float, default=1.5)
        parser.add_argument("--ova-pi", type=float, default=0.5)

        return parser

    @property
    def task(self) -> str:
        return "Testing"

    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.for_what,
            self.checkpoint_hash,
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt


class MTcConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(MTcConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--tau", type=float, default=0.95)
        parser.add_argument("--T", type=float, default=0.5)
        parser.add_argument("--alpha", type=float, default=0.75)
        parser.add_argument("--lambda_u", type=float, default=75)

        return parser

    @property
    def task(self) -> str:
        return "MTC"


class SafeStudentConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SafeStudentConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--T", type=float, default=1, help="temperature parameter for ED"
        )
        parser.add_argument(
            "--lambda-one", type=float, default=1.0, help="coefficient for CBE loss"
        )
        parser.add_argument(
            "--lambda-two", type=float, default=0.01, help="coefficient for UCD loss"
        )
        parser.add_argument("--ema-factor", type=float, default=0.996)
        parser.add_argument("--pretrain-train-split", type=int, default=5, help="")

        return parser

    @property
    def task(self) -> str:
        return "SafeStudent"


class SOFTMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SOFTMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--dist-da-len", type=int, default=128)
        parser.add_argument("--n-sigma", type=float, default=2)
        parser.add_argument("--ema-p", type=float, default=0.999)

        return parser

    @property
    def task(self) -> str:
        return "SOFTMATCH"


class FREEMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(FREEMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--ema-p", type=float, default=0.999)
        parser.add_argument("--ent-loss-ratio", type=float, default=0.01)
        parser.add_argument("--use-quantile", action="store_true")
        parser.add_argument("--clip-thresh", action="store_true")

        return parser

    @property
    def task(self) -> str:
        return "FREEMATCH"


class SIMMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SIMMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--proj-size", type=int, default=128)
        parser.add_argument("--K", type=int, default=2400)
        parser.add_argument("--T", type=float, default=0.1)
        parser.add_argument("--in-loss-ratio", type=float, default=1.0)
        parser.add_argument("--smoothing-alpha", type=float, default=0.9)
        parser.add_argument("--da-len", type=float, default=32)

        return parser

    @property
    def task(self) -> str:
        return "SIMMATCH"


class DEFIXMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(DEFIXMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--T", type=float, default=0.5)
        parser.add_argument("--consis-coef", type=float, default=0.5)

        return parser

    @property
    def task(self) -> str:
        return "DEFIXMATCH"


class REFIXMATCHConfig(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(REFIXMATCHConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            "Linear evaluation of pre-trained model.", add_help=False
        )
        parser.add_argument(
            "--train-augment",
            type=str,
            default="semi",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument(
            "--test-augment",
            type=str,
            default="test",
            choices=("finetune", "test", "semi"),
        )
        parser.add_argument("--p-cutoff", type=float, default=0.95)
        parser.add_argument("--T", type=float, default=0.5)
        parser.add_argument("--consis-coef", type=float, default=0.1)

        return parser

    @property
    def task(self) -> str:
        return "REFIXMATCH"
