import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Funktion zum Parsen des Timestamps
def parse_timestamp(ts):
    return datetime.strptime(ts, '%Y%m%d_%H%M%S%f')

# CSV-Datei einlesen
df = pd.read_csv('rec_fps/control_data.csv')

# Konvertiere Timestamps in datetime-Objekte
df['timestamp'] = df['timestamp'].apply(parse_timestamp)

# Berechne die Dauer zwischen den Frames
df['duration'] = df['timestamp'].diff().dt.total_seconds()

# Drop den ersten Frame, da dessen Dauer NaN ist
df = df.dropna(subset=['duration'])

# Plotten der Dauer und der Kontrollparameter
fig, ax1 = plt.subplots(figsize=(12, 8))

# Dauer der Frames plotten
ax1.set_xlabel('Frame Count')
ax1.set_ylabel('Dauer (Sekunden)', color='tab:blue')
ax1.plot(df['framecount'], df['duration'], marker='o', linestyle='-', color='tab:blue', label='Dauer')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Zweite Y-Achse für die Kontrollparameter
ax2 = ax1.twinx()
ax2.set_ylabel('Kontrollparameter', color='tab:red')
ax2.plot(df['framecount'], df['speed'], marker='x', linestyle='--', color='tab:red', label='Speed')
ax2.plot(df['framecount'], df['angle'], marker='s', linestyle='--', color='tab:orange', label='Angle')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Legende und Titel
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Dauer der Frames und Kontrollparameter')
plt.grid(True)
plt.show()
