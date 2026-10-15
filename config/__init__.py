import configparser
import os


class Config:
    """Load all DLPFC training settings from one INI file."""

    def __init__(self, config_file):
        config_file = os.path.abspath(config_file)
        parser = configparser.ConfigParser(interpolation=None)
        if not parser.read(config_file, encoding="utf-8"):
            raise FileNotFoundError("configuration file not found: {}".format(config_file))

        model = parser["Model_Setup"]
        self.epochs = model.getint("epochs")
        self.lr = model.getfloat("lr")
        self.weight_decay = model.getfloat("weight_decay")
        self.k = model.getint("k")
        self.radius = model.getint("radius")
        self.nhid1 = model.getint("nhid1")
        self.nhid2 = model.getint("nhid2")
        self.dropout = model.getfloat("dropout")
        self.alpha = model.getfloat("alpha")
        self.beta = model.getfloat("beta")
        self.gamma = model.getfloat("gamma")
        self.no_cuda = model.getboolean("no_cuda")
        self.no_seed = model.getboolean("no_seed")
        self.seed = model.getint("seed")

        data = parser["Data_Setting"]
        self.fdim = data.getint("fdim")
        self.datasets = self._get_string_list(data, "datasets")
        self.data_root_template = data.get("data_root_template")
        self.semantic_adj_root = data.get("semantic_adj_root")
        self.data_file_candidates = self._get_string_list(data, "data_file_candidates")

        training = parser["Training_Setup"]
        self.runs_per_dataset = training.getint("runs_per_dataset")
        self.cuda_visible_devices = training.get("cuda_visible_devices")

        model_options = parser["Model_Options"]
        self.fusion_type = model_options.get("fusion_type")
        self.acr_coeff_mode = model_options.get("acr_coeff_mode")
        self.acr_debug = model_options.getboolean("acr_debug")

        experiment = parser["Experiment_Setup"]
        self.acr_norm = experiment.get("acr_norm")
        self.balanced_att_init = experiment.getboolean("balanced_att_init")
        self.avg_fusion_init = experiment.getboolean("avg_fusion_init")
        self.result_root = experiment.get("result_root")
        self.summary_name = experiment.get("summary_name")

    @staticmethod
    def _get_string_list(section, option):
        values = [value.strip() for value in section.get(option).split(",")]
        return [value for value in values if value]
