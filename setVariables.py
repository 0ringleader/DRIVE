#setVariables.py edited 0608 @3:25PM

import configparser
import os

class SetVariables:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.expected_variables = {
            'precalc.py': ['xSize', 'ySize'],
            'neuronalnet.py': ['learning_rate', 'epochs']
            # Füge hier weitere Dateien und deren erwartete Variablen hinzu
        }

    def get_variables(self, section):
        """
        Liest Variablen aus einer angegebenen Sektion der Konfigurationsdatei und überprüft sie.

        Args:
            section (str): Name der Sektion in der Konfigurationsdatei.

        Returns:
            dict: Ein Dictionary der Variablen und deren Werte.
        """
        if section in self.config:
            variables = {key: self._cast_value(value) for key, value in self.config.items(section)}
            self._check_unexpected_variables(section, variables)
            return variables
        else:
            raise KeyError(f"Section '{section}' not found in config file '{self.config_file}'")

    def _cast_value(self, value):
        """
        Versucht, den Wert in einen geeigneten Datentyp zu verwandeln.

        Args:
            value (str): Der ursprüngliche String-Wert aus der Konfigurationsdatei.

        Returns:
            Der Wert in einem entsprechenden Datentyp (int, float, bool, str).
        """
        try:
            if value.isdigit():
                return int(value)
            else:
                try:
                    return float(value)
                except ValueError:
                    if value.lower() in ['true', 'false']:
                        return value.lower() == 'true'
                    return value
        except:
            return value

    def _check_unexpected_variables(self, section, variables):
        """
        Überprüft, ob die Variablen in der Konfigurationsdatei überflüssig sind.

        Args:
            section (str): Name der Sektion in der Konfigurationsdatei.
            variables (dict): Variablen und deren Werte aus der Konfigurationsdatei.
        """
        if section in self.expected_variables:
            expected_vars = self.expected_variables[section]
            for var in variables.keys():
                if var not in expected_vars:
                    print(f"Warning: Unexpected variable '{var}' in section '{section}'")