import mne
import json
from functools import reduce
from pathlib import Path

# --- Configure your session files ---
session_files = ["/Users/podlaskijacek/Documents/HMM-Analysis/data/cleanica/s1_01_pp_cICA_raw.fif",
         "/Users/podlaskijacek/Documents/HMM-Analysis/data/cleanica/s1_02_pp_cICA_raw.fif",
         "/Users/podlaskijacek/Documents/HMM-Analysis/data/cleanica/s1_03_pp_cICA_raw.fif",
         #"/Users/podlaskijacek/Documents/HMM-Analysis/data/cleanica/s2_04_pp_cICA_raw.fif",
         #"/Users/podlaskijacek/Documents/HMM-Analysis/data/cleanica/s2_05_pp_cICA_raw.fif",
         #"/Users/podlaskijacek/Documents/HMM-Analysis/data/cleanica/s2_06_pp_cICA_raw.fif",
         #"/Users/podlaskijacek/Documents/HMM-Analysis/data/cleanica/s2_07_pp_cICA_raw.fif",
]
infos = [mne.io.read_raw_fif(f, preload=False).info for f in session_files]

channel_sets = []
for i, info in enumerate(infos):
    meg_idx = mne.pick_types(info, meg=True, ref_meg=False)
    ch_names = [info['ch_names'][idx] for idx in meg_idx]
    channel_sets.append(set(ch_names))
    print(f"Session {i+1} ({Path(session_files[i]).name}): {len(ch_names)} MEG channels")

common_channels = sorted(reduce(lambda a, b: a & b, channel_sets))
print(f"\nCommon channels across all sessions: {len(common_channels)}")

# Save as .txt for osl-dynamics
txt_path = "common_channels.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(common_channels))
print(f"Saved {txt_path}")

# Save as .json
json_path = "common_channels.json"
with open(json_path, "w") as f:
    json.dump({
        "n_common_channels": len(common_channels),
        "common_channels": common_channels,
        "session_info": [
            {
                "file": Path(fp).name,
                "n_channels": len(cs),
                "n_dropped": len(cs - set(common_channels)),
                "dropped": sorted(cs - set(common_channels)),
            }
            for fp, cs in zip(session_files, channel_sets)
        ]
    }, f, indent=2)
print(f"Saved {json_path}")