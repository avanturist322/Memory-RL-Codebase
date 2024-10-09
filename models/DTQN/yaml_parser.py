from ruamel.yaml import YAML
import yaml


class YamlParser:
    """The YamlParser parses a yaml file containing parameters for the environment, model, evaulation, and trainer.
    The data is parsed during initialization.
    Retrieve the parameters using the get_config function.
    The data can be accessed like:
    parser.get_config()["environment"]["name"]
    """


    def __init__(self, path):
        """Loads and prepares the specified config file.
        
        Arguments:
            path {str} -- Yaml file path to the to be loaded config file.
        """
        # Load the config file
        with open(path, 'r') as file:
            self._config = yaml.safe_load(file)

        # yaml = YAML()
        # yaml_args = yaml.load_all(stream)
        
        # # Final contents of the config file will be added to a dictionary
        # self._config = {}

        # # Prepare data
        # for data in yaml_args:
        #     self._config = dict(data)

    def get_config(self):
        """ 
        Returns:
            {dict} -- Nested dictionary that contains configs for the environment, model, evaluation and trainer.
        """
        return self._config