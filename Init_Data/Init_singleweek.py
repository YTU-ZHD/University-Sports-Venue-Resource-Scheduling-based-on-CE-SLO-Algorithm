import pandas as pd
import numpy as np

seed = 123
np.random.seed(seed)

counts = {
    'Hard Game': 8,
    'Soft Game': 12,
    'Teaching': 60,
    'Student Club': 40,
    'Free Exercise': 50
}

allowed_days = {
    'Hard Game': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    'Soft Game': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    'Teaching': ['Monday','Tuesday','Wednesday','Thursday','Friday'],
    'Student Club': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    'Free Exercise': ['Saturday','Sunday']
}

durations = {
    'Hard Game': lambda: np.random.randint(3, 5),
    'Soft Game': lambda: np.random.randint(3, 5),
    'Teaching': lambda: 2,
    'Student Club': lambda: np.random.randint(1, 4),
    'Free Exercise': lambda: 1
}

records = []
request_id = 1
T = 12
for rtype, cnt in counts.items():
    for _ in range(cnt):
        day = np.random.choice(allowed_days[rtype])
        duration = durations[rtype]()
        start_slot = np.random.randint(1, T - duration + 2)
        records.append({
            'RequestID': request_id,
            'Type': rtype,
            'DesiredDay': day,
            'DesiredSlot': start_slot,
            'Duration': duration
        })
        request_id += 1

df = pd.DataFrame(records)

df.to_csv(f"DataSet_weeks1_{seed}.csv", index=False)

print("Corrected Mock Request Dataset (Top 10)")
print(df.head(10))

print("\nNumber of Requests by Type:")
print(df['Type'].value_counts())
