"""
Utility functions for atomic, compressed saving and loading of structured data.

Supports pickle, gzip, lz4, CSV, and sparse matrix formats. Used throughout
Apriori for consistent I/O behavior and compression handling.
"""
from pathlib import Path
import os
import pickle
import gzip
import lz4.frame
import tempfile
import pandas as pd
from scipy.sparse import load_npz


def save_to_pickle(pkl_object, file_path, gz_compression=False, lz4_compression=False, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Atomically saves a Python object to a pickle file with optional compression.

    Parameters
    ----------
    pkl_object : Any
        The Python object to serialize.

    file_path : str or Path
        The path to save the pickle file. If compression is enabled, the correct
        extension (.gz or .lz4) will be appended if not already present.

    gz_compression : bool, default=False
        If True, compress the pickle file with gzip.

    lz4_compression : bool, default=False
        If True, compress the pickle file with lz4.

    protocol : int, default=pickle.HIGHEST_PROTOCOL
        The pickle protocol to use when serializing the object.

    Raises
    ------
    ValueError
        If both `gz_compression` and `lz4_compression` are True.
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)

    if gz_compression and lz4_compression:
        raise ValueError("Cannot use both gzip and lz4 compression.")

    if gz_compression and not file_path.endswith(".gz"):
        file_path += ".gz"
    elif lz4_compression and not file_path.endswith(".lz4"):
        file_path += ".lz4"

    # Optional strict extension check (uncomment if desired)
    # if gz_compression and file_path.endswith(".lz4"):
    #     raise ValueError("File path ends with .lz4 but gzip compression was selected.")
    # if lz4_compression and file_path.endswith(".gz"):
    #     raise ValueError("File path ends with .gz but lz4 compression was selected.")

    open_fn = (
        lz4.frame.open if lz4_compression else
        gzip.open if gz_compression else
        open
    )

    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    fd, temp_path = tempfile.mkstemp(prefix=f".tmp_{base_name}", dir=dir_name)
    os.close(fd)

    try:
        with open_fn(temp_path, 'wb') as handle:
            pickle.dump(pkl_object, handle, protocol=protocol)
        os.replace(temp_path, file_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def load_from_pickle(file_path: str):
    """
    Loads a Python object from a pickle file based on file extension.

    Parameters
    ----------
    file_path : str
        The path to load the pickle file from. Supported extensions:
        - .pkl       (uncompressed)
        - .pkl.gz    (gzip-compressed)
        - .pkl.lz4   (lz4-compressed)
        - .gz        (gzip-compressed, legacy)
        - .lz4       (lz4-compressed, legacy)

    Returns
    -------
    Any or None
        The deserialized Python object, or None if the file does not exist.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)
    
    if not os.path.isfile(file_path):
        return None

    if file_path.endswith(".pkl.lz4") or file_path.endswith(".lz4"):
        open_fn = lz4.frame.open
    elif file_path.endswith(".pkl.gz") or file_path.endswith(".gz"):
        open_fn = gzip.open
    elif file_path.endswith(".pkl"):
        open_fn = open
    else:
        raise ValueError(f"Unsupported file extension for: {file_path}")

    with open_fn(file_path, "rb") as handle:
        return pickle.load(handle)

def load_from_disk(file_path: str):
    """
    Loads a structured object from disk based on file extension.

    Supports CSV, compressed pickles, and sparse matrix `.npz` files.

    Parameters
    ----------
    file_path : str
        Path to the file to load. Supported extensions:
        - .csv, .csv.gz → loaded as pandas DataFrame
        - .pkl, .pkl.gz, .pkl.lz4, .gz, .lz4 → loaded via pickle
        - .npz → loaded as scipy.sparse.csr_matrix

    Returns
    -------
    object
        The loaded object (e.g., DataFrame, dict, sparse matrix)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the file extension is unsupported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Missing required file: {file_path}")

    print(f"[load_from_disk] Loading: {file_path}")

    path_lower = str(file_path).lower()

    if path_lower.endswith(".csv") or path_lower.endswith(".csv.gz"):
        return pd.read_csv(file_path, low_memory=False)
    elif path_lower.endswith(".npz"):
        return load_npz(file_path)
    elif (
        path_lower.endswith(".pkl")
        or path_lower.endswith(".pkl.gz")
        or path_lower.endswith(".pkl.lz4")
        or path_lower.endswith(".lz4")
        or path_lower.endswith(".gz")  # legacy gzip pickle
    ):
        return load_from_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
