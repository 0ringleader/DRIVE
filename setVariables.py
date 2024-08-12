#setVariables.py edited 1208 @2:20PM by Sven

import configparser

class SetVariables:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.expected_variables = {
            'precalc.py': ['xsize', 'ysize', 'edge_strength', 'noise_h', 'noise_hcolor', 'noise_templatewindowsize',
                           'noise_searchwindowsize', 'canny_threshold1', 'canny_threshold2', 'clahe_cliplimit',
                           'clahe_tilegridsize', 'analyzeimages', 'overrideimages', 'n_clusters'],
            'neuronalnet.py': ['learning_rate', 'epochs', 'batch_size', 'validation_split', 'l1_reg', 'l2_reg',
                               'dropout_rate', 'data_dir', 'xSize', 'ySize', 'printConsole', 'considerRecentControls']
        }

    def get_variables(self, section):
        variables = {}
        if section in self.config:
            variables.update({key.lower(): self._cast_value(value) for key, value in self.config.items(section)})

        if section == 'neuronalnet.py' and 'precalc.py' in self.config:
            variables['xsize'] = self._cast_value(self.config['precalc.py']['xsize'])
            variables['ysize'] = self._cast_value(self.config['precalc.py']['ysize'])

        self._check_unexpected_variables(section, variables)
        print(f"Loaded variables for section {section}: {variables}")  # Debug output
        return variables

    def _cast_value(self, value):
        value = value.strip().strip('"')  # Remove quotation marks and whitespace
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _check_unexpected_variables(self, section, variables):
        if section in self.expected_variables:
            expected_vars = self.expected_variables[section]
            for var in variables.keys():
                if var not in expected_vars:
                    print(f"Warning: Unexpected variable '{var}' in section '{section}'")

def replySetVariables(section):
    config = SetVariables('config.ini')
    try:
        variables = config.get_variables(section)
        for var, value in variables.items():
            print(f"{var} = {value}")
        print(f"All expected variables for section '{section}' loaded successfully.")
    except KeyError as e:
        print(str(e))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
