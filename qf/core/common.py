from pathlib import Path


def data_directory(name='data'):
    """
    Return the directory that contains the data.
    
    We assume that the data folder is locate in a parent directory of this file and named 'data'.
    If your setup is different, you will need to change this method.
    """
    dataDir = Path(__file__).resolve().parent
    while not list(dataDir.rglob('data')):
        dataDir = dataDir.parent
    found = [d for d in dataDir.rglob('data') if d.is_dir()]
    if not found:
        raise Exception(f'Cannot find data directory with name {name} along the path of your source files')
    return found[0]
    
    
def data_file(file_name, data_dir_name='data'):
    """
    Return the path to a data file in the data directory.
    
    This is a convenience function to get the full path to a file in the data directory.
    """
    return data_directory(data_dir_name) / file_name  