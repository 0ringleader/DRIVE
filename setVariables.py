#setVariables.py edited 0608 @10:30PM by Sven
#not finished yet!

import configparser

class SetVariables:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.expected_variables = {
            'precalc.py': ['xSize', 'ySize', 'edge_strength', 'noise_h', 'noise_hColor', 'noise_templateWindowSize', 'noise_searchWindowSize', 'canny_threshold1', 'canny_threshold2', 'clahe_clipLimit', 'clahe_tileGridSize'],
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
            print(f"Loaded variables for section {section}: {variables}")  # Debug-Ausgabe
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
            if value.replace('.', '', 1).isdigit():
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            elif value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            else:
                return value
        except ValueError:
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

def replySetVariables(section):                                                             #Diese Funktion soll nur überprüfen, ob alle Variablen geladen wurden (debug-funktion)
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

