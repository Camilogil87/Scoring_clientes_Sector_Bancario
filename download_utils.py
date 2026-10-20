#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import requests
import tqdm_utils
import gzip
from pathlib import Path


def download_file(url, file_path):
    fpath = Path(file_path)
    folder = fpath.parent


    if not folder.exists():
        os.makedirs(folder)

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    try:
        with open(file_path, 'wb', buffering=16 * 1024 * 1024) as f:
            bar = tqdm_utils.tqdm_notebook_failsafe(total=total_size, unit='B', unit_scale=True)
            bar.set_description(os.path.split(file_path)[-1])
            for chunk in r.iter_content(32 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
            bar.close()
    except Exception:
        print("Download failed")
    finally:
        if os.path.getsize(file_path) != total_size:
            os.remove(file_path)
            print("Removed incomplete download")


def unzip_file(path, remove_orig=False):
    assert path.endswith('.gz')
    # rm extension
    path_out = '.'.join(path.split('.')[:-1])
    # unzip
    with gzip.open(path, 'rb') as f_in:
        with open(path_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if remove_orig:
        os.remove(path)


def download_embeddings_challenge(force=False):
    url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    url2 = "https://figshare.com/ndownloader/files/10798046"
    path = "../embeddings/GoogleNews-vectors-negative300.bin.gz"
    path2 = "../embeddings/GoogleNews-vectors-negative300.bin"

    fname = os.path.split(path)[-1]
    fname2 = os.path.split(path2)[-1]

    try:
        if not os.path.isfile(path) and not os.path.isfile(path[:-3]):
            print(f"Downloading {fname} (1.5G) for you, it will take a while...")
            download_file(url,
                          path)
        if not os.path.isfile(path[:-3]):
            print('Unzipping file...')
            unzip_file(path, remove_orig=True)
        else:
            print('File already downloaded!')
    except:
        print("Link is not available, trying with a second link...")
        if not os.path.isfile(path2):
            print(f"Downloading {fname2} (3.6GB) for you, it will take a while...")
            download_file(url2,
                          path2)
        else:
            print('File already downloaded!')

    print('Done!')


def download_embeddings_worksheet(force=False):
    url = 'http://dcc.uchile.cl/~jperez/word-embeddings/fasttext-sbwc.100k.vec.gz'
    path = "../embeddings/fasttext-sbwc.100k.vec.gz"

    fname = os.path.split(path)[-1]
    
    print(f"Downloading {fname} (98M) for you, it will take a while...")
    if not os.path.isfile(path) and not os.path.isfile(path[:-3]):
        download_file(url,
                      path)
    if not os.path.isfile(path[:-3]):
        print('Unzipping file...')
        unzip_file(path, remove_orig=True)
    else:
        print('File already downloaded!')
    print('Done!')
