#Script equivalent to "preprocessing.ipynb", but can be used as a function

from preproc_funcs import detect_bad_channels, detect_bad_segments, plot_spectra, plot_freqbands, plot_channel_time_series
from pathlib import Path
import mne
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
def preprocessing(dir, subject, sessions):

    raw_file = f"{dir}/data/raw/s{subject}_{string[i]}_raw.fif"
    #raw = mne.io.read_raw_ctf(raw_file, preload=True) <- for .ds folders and .meg4 datafiles
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    
    plot_channel_time_series(raw, savebase=f"{dir}/plots/raw/channel_time_series/s{subject}_{string[i]}_raw", exclude_bads=False)
    plot_spectra(raw, savebase=f"{dir}/plots/raw/spectra/s{subject}_{string[i]}_raw")
    plot_freqbands(raw, savebase=f"{dir}/plots/raw/freqbands/s{subject}_{string[i]}_raw")
    
    raw = raw.filter(l_freq=1, h_freq=80, method="iir", iir_params={"order": 5, "ftype": "butter"})
    raw = raw.notch_filter([50, 100])
    raw = raw.resample(sfreq=250)
    
    plot_channel_time_series(raw, savebase=f"{dir}/plots/pp/channel_time_series/s{subject}_{string[i]}_filter", exclude_bads=False)
    plot_spectra(raw, savebase=f"{dir}/plots/pp/spectra/s{subject}_{string[i]}_filter")
    plot_freqbands(raw, savebase=f"{dir}/plots/pp/freqbands/s{subject}_{string[i]}_filter")
    
    raw = detect_bad_segments(raw, picks="mag", significance_level=0.1)
    raw = detect_bad_segments(raw, picks="mag", significance_level=0.1, mode="diff")
    raw = detect_bad_channels(raw, picks='mag', significance_level=0.1)
    
    plot_channel_time_series(raw, savebase=f"{dir}/plots/pp/channel_time_series/s{subject}_{string[i]}_filter+bad", exclude_bads=False)
    plot_spectra(raw, savebase=f"{dir}/plots/pp/spectra/s{subject}_{string[i]}_filter+bad")
    plot_freqbands(raw, savebase=f"{dir}/plots/pp/freqbands/s{subject}_{string[i]}_filter+bad")
    
    preproc_file = Path(f"{dir}/data/pp/s{subject}_{string[i]}_pp_raw.fif") #.fif file format
    preproc_file.parent.mkdir(parents=True, exist_ok=True)
    raw.save(preproc_file, overwrite=True)


string = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
dir = "/Volumes/ssd X9/HMM_data"
subject=2
sessions=1,15
for i in range(sessions[0],(sessions[1]+1)):
    preprocessing(dir, subject, sessions)


 

    
    
