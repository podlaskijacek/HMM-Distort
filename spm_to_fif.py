"""Convert an SPM MEEG .mat/.dat file to MNE-Python .fif format."""

import numpy as np
import scipy.io
import mne
from mne.io.constants import FIFF


# SPM channel type -> MNE channel type
TYPE_MAP = {
    "REFMAG": "ref_meg",
    "REFGRAD": "ref_meg",
    "REFPLANAR": "ref_meg",
    "EEG": "eeg",
    "ECG": "ecg",
    "EMG": "emg",
    "LFP": "bio",
    "PHYS": "bio",
    "ILAM": "bio",
    "SRC": "dipole",
    "EOG": "eog",
    "VEOG": "eog",
    "HEOG": "eog",
    "MEG": "mag",
    "MEGMAG": "mag",
    "MEGCOMB": "misc",
    "MEGGRAD": "mag",
    "MEGPLANAR": "grad",
    "TRIG": "stim",
    "Other": "misc",
}

# SPM unit string -> multiplier to convert to SI (T for MEG, V for EEG)
UNIT_MULTIPLIERS = {
    "Nan": 1,
    "mT": 1e-3,
    "nT": 1e-9,
    "pT": 1e-12,
    "fT": 1e-15,
    "T": 1,
    "V": 1,
    "mV": 1e-3,
    "uV": 1e-6,
    "unknown": 1,
    "s": 1,
}

# NIFTI dtype codes -> numpy dtypes
NIFTI_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def spm_to_fif(mat_path, dat_path, output_path):
    """Convert SPM MEEG .mat/.dat files to MNE .fif format.

    Parameters
    ----------
    mat_path : str
        Path to the SPM .mat file.
    dat_path : str
        Path to the SPM .dat file (raw binary data).
    output_path : str
        Path for the output .fif file.
    """
    D = scipy.io.loadmat(mat_path, squeeze_me=True)["D"]

    # -- Dimensions --
    sfreq = float(D["Fsample"].item())
    data_dim = D["data"].item()["dim"].item()
    n_channels = int(data_dim[0])
    n_samples = int(data_dim[1])

    # -- Channel info --
    channels = D["channels"].item()
    ch_labels = list(channels["label"])
    spm_types = list(channels["type"])
    spm_units = list(channels["units"])
    assert len(ch_labels) == n_channels

    mne_types = [TYPE_MAP.get(t, "misc") for t in spm_types]
    info = mne.create_info(ch_names=ch_labels, sfreq=sfreq, ch_types=mne_types)

    # -- Load raw data from .dat --
    spm_dtype = int(D["data"].item()["dtype"].item())
    np_dtype = NIFTI_DTYPES.get(spm_dtype, np.float32)
    data = (
        np.fromfile(dat_path, dtype=np_dtype)
        .reshape(n_channels, n_samples, order="F")
        .astype(np.float64)
    )

    # -- Unit conversion (e.g. fT -> T) --
    unit_mul = np.array([UNIT_MULTIPLIERS.get(u, 1) for u in spm_units])
    data *= unit_mul[:, np.newaxis]

    # -- MEG sensor positions --
    try:
        meg_sens = D["sensors"].item()["meg"].item()
        meg_labels = list(meg_sens["label"].item())
        chanpos = meg_sens["chanpos"].item()
        chanori = meg_sens["chanori"].item()

        # mm -> m
        k_unit = 1e-3 if str(meg_sens["unit"].item()) == "mm" else 1

        # Rotation matrix: 90 degrees anti-clockwise viewed from above
        # (CTF head coords -> MNE head coords)
        # new_x = -old_y, new_y = old_x, new_z = old_z
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

        # Build label -> sensor index map
        meg_label_to_idx = {lab: i for i, lab in enumerate(meg_labels)}

        i_meg = 0
        for idx, name in enumerate(ch_labels):
            ch = info["chs"][idx]
            if ch["kind"] == FIFF.FIFFV_MEG_CH:
                if name in meg_label_to_idx:
                    si = meg_label_to_idx[name]
                    pos = rot @ (chanpos[si] * k_unit)
                    ori = rot @ chanori[si]
                    ch["loc"] = np.concatenate((pos, ori, np.zeros(6)))
                ch["unit"] = FIFF.FIFF_UNIT_T
                ch["unit_mul"] = FIFF.FIFF_UNITM_NONE
                ch["range"] = 1.0
                ch["cal"] = 1.0
                ch["coord_frame"] = FIFF.FIFFV_COORD_DEVICE
                i_meg += 1

        # Handle tra matrix
        tra = meg_sens["tra"].item()
        if tra.shape[0] != tra.shape[1]:
            tra_path = dat_path.replace(".dat", "_tra_matrix.tsv")
            np.savetxt(tra_path, tra, delimiter="\t")
            print(f"  tra matrix is non-square {tra.shape}, saved to {tra_path}")
        else:
            rank_tra = np.linalg.matrix_rank(tra)
            _, _, V = np.linalg.svd(tra, full_matrices=False)
            V_real = np.conj(V.T)
            proj_vec = V_real[:, rank_tra:]
            if proj_vec.shape[1] > 0:
                for ii in range(proj_vec.shape[1]):
                    proj_data = dict(
                        col_names=list(meg_labels),
                        row_names=None,
                        data=proj_vec[:, ii].reshape(1, -1),
                        ncol=len(meg_labels),
                        nrow=1,
                    )
                    info["projs"].append(
                        mne.Projection(
                            active=True,
                            data=proj_data,
                            desc="from SPM tra matrix",
                            kind=1,
                            explained_var=None,
                        )
                    )
    except (ValueError, KeyError):
        print("  No MEG sensor info found")

    # -- EEG sensor positions --
    try:
        eeg_sens = D["sensors"].item()["eeg"].item()
        eeg_labels = list(eeg_sens["label"].item())
        eeg_chanpos = eeg_sens["chanpos"].item()
        eeg_label_to_idx = {lab: i for i, lab in enumerate(eeg_labels)}

        for idx, name in enumerate(ch_labels):
            ch = info["chs"][idx]
            if ch["kind"] == FIFF.FIFFV_EEG_CH:
                if name in eeg_label_to_idx:
                    si = eeg_label_to_idx[name]
                    ch["loc"] = np.concatenate((eeg_chanpos[si], np.zeros(9)))
                ch["unit"] = FIFF.FIFF_UNIT_V
                ch["unit_mul"] = FIFF.FIFF_UNITM_NONE
                ch["range"] = 1.0
                ch["cal"] = 1.0
                ch["coord_frame"] = FIFF.FIFFV_COORD_HEAD
    except (ValueError, KeyError):
        pass

    # -- Create RawArray --
    raw = mne.io.RawArray(data, info)

    # -- Bad channels --
    bad_flags = channels["bad"]
    bad_channels = [ch_labels[i] for i in range(n_channels) if int(bad_flags[i]) == 1]
    raw.info["bads"] = bad_channels

    # -- Save --
    raw.save(output_path, overwrite=True)
    print(f"Saved {output_path}")
    print(f"  {n_channels} channels, {n_samples} samples, {sfreq} Hz")
    print(f"  Bad channels: {bad_channels if bad_channels else 'none'}")


string = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]

dir = "/Volumes/ssd X9/HMM_data"
subject=2
sessions=1,15

for i in range(sessions[0],(sessions[1])+1):

    # Input files
    mat_file = f"{dir}/data/mat_dat/sub{subject}_run{string[i]}.mat"
    dat_file = mat_file.replace(".mat", ".dat")

    # Output file
    output_file = f"{dir}/data/raw/s{subject}_{string[i]}_raw.fif"

    # Convert SPM to fif
    spm_to_fif(mat_file, dat_file, output_file)
