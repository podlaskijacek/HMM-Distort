import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pathlib import Path

def ICAc(dir, subject, sessions):
    sample_data_raw_file = f"{dir}/data/pp/s{subject}_{string[i]}_preproc_raw.fif"
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
    
    ica = ICA(
        n_components=80,
        max_iter="auto",
        method="infomax",
        random_state=97,
        fit_params=dict(extended=True),
    )
    
    ica.fit(raw)


    ic_labels = label_components(raw, ica, method="megnet")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain/other"]]

    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)

    ica_file = Path(f"{dir}/data/ica/s{subject}_{string[i]}_pp_ICA_raw.fif") #.fif file format
    raw.save(ica_file, overwrite=True)
    
    icaclean_file = Path(f"{dir}/data/cleanica/s{subject}_{string[i]}_pp_cICA_raw.fif") #.fif file format
    reconst_raw.save(icaclean_file, overwrite=True)
    

string = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
dir = "/Users/podlaskijacek/Documents/HMM-Analysis"
subject=2
sessions=1,15
for i in range(sessions[0],(sessions[1]+1)):
    ICAc(dir, subject, sessions)

