import json
from pathlib import Path
from typing import Union, List, Any
from argparse import Namespace


class DefaultConfig:
    """Defines the default BIONIC config parameters.
    """

    # Default config parameters used unless specified by user
    _defaults = {
        "names": None,  # Filenames of input networks
        "out_name": None,  # Name of output feature, model and network weight files
        "delimiter": " ",  # Delimiter for network input files
        "epochs": 3000,  # Number of training epochs
        "batch_size": 2048,  # Number of genes/proteins in each batch
        "sample_size": 0,  # Number of networks to batch at once (0 will batch all networks)
        "learning_rate": 0.0005,  # Adam optimizer learning rate
        "embedding_size": 512,  # Dimensionality of output integrated features
        "svd_dim": 0,  # Dimensionality of network SVD approximation (0 will not perform SVD)
        # "initialization": "xavier",  # Method used to initialize BIONIC weights
        # "gat_shapes": {
        #     "dimension": 64,  # Dimension of each GAT layer
        #     "n_heads": 10,  # Number of attention heads for each GAT layer
        #     "n_layers": 2,  # Number of GAT layers for each input network
        # },
        "save_network_scales": False,  # Whether to save internal learned network feature scaling
        "save_model": True,  # Whether to save the trained model or not
        "load_pretrained_model": False,  # Whether to load a pretrained model TODO
        "use_tensorboard": False,  # Whether to output tensorboard data
        "plot_loss": True,  # Whether to plot loss curves
    }

    # Required parameters not specified in `_defaults`
    _required_params = {"names"}

    # Parameters that should be cast to `Path` type
    _path_args = {"names", "out_name"}

    def __init__(self, config: dict):

        self.config = config

        # Make sure all path strings are mapped to `Path`s
        for arg in DefaultConfig._path_args:
            if isinstance(self.config[arg], list):
                self.config[arg] = [
                    Path(path_string).expanduser() for path_string in self.config[arg]
                ]
            else:
                self.config[arg] = Path(self.config[arg]).expanduser()


class ConfigParser(DefaultConfig):
    def __init__(self, config: Union[Path, dict]):
        """Parses and validates a user supplied config

        Args:
            config (Union[Path, dict]): Name of config file or preloaded config
                dictionary.
        """

        self.config = config
        super().__init__(self.config)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Union[Path, dict]) -> None:

        # Check if `config` is already loaded and validate `out_name` exists if so
        if isinstance(config, dict) and "out_name" not in config:
            raise ValueError(
                "Output file name `out_name` must be provided in `config` if "
                "`config` is provided as a dictionary"
            )

        # If `config` is a `Path`, load and set `out_name` parameter if not specified
        if isinstance(config, Path):
            config_path = config
            with config_path.open() as f:
                config = json.load(f)  # config is now a dictionary if it was previously a `Path`

            if "out_name" not in config:
                config["out_name"] = config_path.parent / config_path.stem

        # Validate required parameters are present in `config`
        required_params = DefaultConfig._required_params
        if len(required_params.intersection(set(config.keys()))) != len(required_params):
            missing_params = "`, `".join(DefaultConfig._required_params - set(config.keys()))
            raise ValueError(
                f"Required parameter(s) `{missing_params}` not found in provided " "config file."
            )

        self._config = config

    def _resolve_asterisk_path(self, path: Path) -> List[Path]:
        directory = path.parent
        return [p for p in directory.iterdir() if p.is_dir()]

    def _get_param(self, param: str, default: Any) -> Any:
        if param in self.config:
            value = self.config[param]

            # Handle `Path` versions of "names" parameter
            if param == "names" and isinstance(value, Path):
                if value.stem == "*":
                    return self._resolve_asterisk_path(value)
                else:
                    return [value]  # Wrap path in a list to ensure compatibility

            return value

        else:
            return default

    def parse(self) -> Namespace:
        """Parses config file.

        Overrides config defaults with user provided params and returns them
        namespaced.

        Returns:
            Namespace: A Namespace object containing parsed BIONIC parameters.
        """

        parsed_params = {
            param: self._get_param(param, default)
            for param, default in DefaultConfig._defaults.items()
        }
        for key, value in self.config.items():
            if key not in parsed_params:
                parsed_params[key] = value
        namespace = Namespace(**parsed_params)
        return namespace
