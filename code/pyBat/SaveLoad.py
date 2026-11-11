import pickle
import os
import natsort
import sys

def find_pickles(folder, return_basename=False, extension='.pck'):
    pickle_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                if return_basename:
                    pickle_files.append(file)
                else:
                    pickle_files.append(os.path.join(root, file))
    pickle_files = natsort.natsorted(pickle_files)
    return pickle_files

def get_pickle_metadata(file):
    """
    Retrieves metadata from a pickle file.
    If no metadata is found, returns None.
    """
    with open(file, 'rb') as f:
        try:
            # Attempt to load the first object as metadata
            metadata = pickle.load(f)
            if isinstance(metadata, dict) and "python_version" in metadata:
                return metadata
        except (EOFError, ValueError, pickle.UnpicklingError):
            pass  # No metadata or invalid metadata
    return None

def pickle_save(file, obj):
    """
    Saves an object to a pickle file, including metadata about Python and pickle versions.
    """
    metadata = {
        "python_version": sys.version,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
    }
    with open(file, 'wb') as f:
        # Save metadata first
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save the actual object
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file):
    """
    Loads an object from a pickle file.
    If metadata is present, it is ignored, and only the object is returned.
    """
    with open(file, 'rb') as f:
        try:
            # Try loading metadata
            metadata = pickle.load(f)
            # Ensure metadata is valid; skip it if not
            if not isinstance(metadata, dict) or "python_version" not in metadata:
                raise ValueError("Invalid metadata; treating file as object-only pickle.")
        except (EOFError, ValueError):
            # No metadata present, reset the file pointer
            f.seek(0)
        # Load the actual object
        obj = pickle.load(f)
    return obj





