import yaml
def load_config(config_path):
    """
    Load configuration from a given file path.

    Args:
        config_path (str): The path to the configuration file.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

