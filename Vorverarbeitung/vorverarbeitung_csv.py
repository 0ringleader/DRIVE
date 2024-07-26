import os
import shutil
import pandas as pd

# Anzahl der gespeicherten Steuerungsdaten
NUM_PREVIOUS = 100  # Anzahl der vergangenen Steuerungsdaten
NUM_NEXT = 100      # Anzahl der zukünftigen Steuerungsdaten

def copy_csv(src_path, dst_path):
    """
    Kopiert eine CSV-Datei von src_path nach dst_path.
    """
    try:
        shutil.copy(src_path, dst_path)
        print(f"Datei erfolgreich von {src_path} nach {dst_path} kopiert.")
    except Exception as e:
        print(f"Fehler beim Kopieren der Datei: {e}")

def process_csv(file_path, output_dir):
    """
    Liest die CSV-Datei und verarbeitet die Steuerungsdaten.
    Fügt die letzten NUM_PREVIOUS und nächsten NUM_NEXT Steuerungsdaten zu jedem Datenpunkt hinzu.
    Speichert die verarbeitete Datei im angegebenen Output-Verzeichnis.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"CSV-Datei erfolgreich gelesen: {file_path}")

        # Erstellen von leeren Listen für die neuen Spalten
        past_speed = []
        past_angle = []
        future_speed = []
        future_angle = []

        # Initialisieren der Listen mit den ersten NUM_PREVIOUS und NUM_NEXT Steuerungsdaten
        initial_speeds = [0] * NUM_PREVIOUS
        initial_angles = [0] * NUM_PREVIOUS
        initial_future_speeds = [0] * NUM_NEXT
        initial_future_angles = [0] * NUM_NEXT

        for i in range(len(df)):
            if i < NUM_PREVIOUS:
                past_speed.append(initial_speeds[:i] + df['speed'][:i].tolist())
                past_angle.append(initial_angles[:i] + df['angle'][:i].tolist())
            else:
                past_speed.append(df['speed'][i-NUM_PREVIOUS:i].tolist())
                past_angle.append(df['angle'][i-NUM_PREVIOUS:i].tolist())
            
            if i + NUM_NEXT >= len(df):
                future_speed.append(df['speed'][i+1:].tolist() + initial_future_speeds[:(i+NUM_NEXT-len(df))])
                future_angle.append(df['angle'][i+1:].tolist() + initial_future_angles[:(i+NUM_NEXT-len(df))])
            else:
                future_speed.append(df['speed'][i+1:i+NUM_NEXT+1].tolist())
                future_angle.append(df['angle'][i+1:i+NUM_NEXT+1].tolist())

        # Hinzufügen der neuen Spalten zum DataFrame
        df['past_speed'] = past_speed
        df['past_angle'] = past_angle
        df['future_speed'] = future_speed
        df['future_angle'] = future_angle

        # Beispielausgabe der ersten 5 Zeilen mit den neuen Spalten
        print(df.head())

        # Erstellen des Output-Pfads und Speichern der verarbeiteten CSV-Datei
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, "control_data_processed.csv")
        df.to_csv(output_file_path, index=False)
        print(f"Verarbeitete Datei gespeichert unter: {output_file_path}")

    except Exception as e:
        print(f"Fehler beim Verarbeiten der CSV-Datei: {e}")

def process_all_csv_in_dataset(dataset_dir, output_base_dir):
    """
    Durchläuft alle Ordner im dataset_dir und verarbeitet alle CSV-Dateien.
    Speichert die Ergebnisse in den entsprechenden Ordnern im output_base_dir.
    """
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("control_data.csv"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, dataset_dir)
                output_dir = os.path.join(output_base_dir, relative_path)
                process_csv(file_path, output_dir)

if __name__ == "__main__":
    # Definiere den Pfad zum Datensatz und zum Output-Verzeichnis
    dataset_dir = "datensatz"
    output_base_dir = "C:/_Code/TH/DRIVE/PreCalcOut"

    # Verarbeite alle CSV-Dateien im Datensatz
    process_all_csv_in_dataset(dataset_dir, output_base_dir)
