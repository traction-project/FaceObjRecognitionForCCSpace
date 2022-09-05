# Copyright (c) 2022 Kevin McGuinness <kevin.mcguinness@dcu.ie> Anderson Simiscuka <anderson.simiscuka2@mail.dcu.ie>
import os
import os.path
import requests
import tqdm
import tarfile


def findfiles(path, suffix):
    """
    Finds all files in the folder specified by `path` that have the given 
    suffix.
    """
    allfiles = []
    for file in os.listdir(path):
        if file.endswith(suffix):
            allfiles.append(os.path.join(path, file))
    allfiles.sort()
    return allfiles


def download_file(url, target=None, path=None, progress=False, chunk_size=8192, 
                  skip_existing=False):
    if target is None:
        target = url.split('/')[-1]
    if path is not None:
        target = os.path.join(path, target)

    if skip_existing and os.path.isfile(target):
        return target

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        if progress:
            nbytes = int(r.headers['content-length'])
            nchunks = nbytes / chunk_size
        with open(target, 'wb') as f:
            content_iter = r.iter_content(chunk_size=chunk_size)
            if progress:
                content_iter = tqdm.tqdm(content_iter, total=nchunks)
            for chunk in content_iter:
                f.write(chunk)
    return target


def extract_tar(filename, path='.'):
    f = tarfile.open(filename, 'r')
    try:
        f.extractall(path=path)
    finally:
        f.close()