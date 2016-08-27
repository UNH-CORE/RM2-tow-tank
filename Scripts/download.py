#!/usr/bin/env python
"""Download (or re-download) all raw data."""

from __future__ import division, print_function
import requests
import tarfile
import json
import os
import re
import wget
import sys

article = "1302029"
BASE_URL = "https://api.figshare.com/v2/{endpoint}"


def get_article_details():
    endpoint = "articles/{}".format(article)
    resp = requests.get(BASE_URL.format(endpoint=endpoint))
    return json.loads(resp.content.decode())


def get_uploaded_files():
    """Return a list of dictionaries describing each file."""
    return get_article_details()["files"]


def get_uploaded_filenames():
    flist = get_uploaded_files()
    return [f["name"] for f in flist if f["name"] != "README.md"]


def get_remote_url(filename):
    """Return remote URL for downloading file."""
    files = get_uploaded_files()
    base = "https://ndownloader.figshare.com/files/{id}"
    remote_urls = {f["name"]: base.format(id=f["id"]) for f in files}
    return remote_urls[filename]


def remote_fname_to_local(remote_name):
    """Convert remote file name to local destination path."""
    base_dir = os.path.join("Data", "Raw")
    split_name = remote_name.split("_")
    fname = split_name[-1]
    run = split_name[-2]
    section = "_".join(split_name[:-2])
    return os.path.join(base_dir, section, run, fname)


def download_file(filename, destination):
    """Download remote file using the `wget` Python module."""
    destdir = os.path.split(destination)[0]
    if not os.path.isdir(destdir):
        os.makedirs(destdir)
    url = get_remote_url(filename)
    wget.download(url, out=destination)


def test_remote_name_to_local():
    fname="Perf-tsr_0-b_7_metadata.json"
    fpath = os.path.join("Data", "Raw", "Perf-tsr_0-b", "7", "metadata.json")
    assert remote_fname_to_local(fname) == fpath


def test_download_file(fname="Perf-tsr_0-b_7_metadata.json"):
    dest = remote_fname_to_local(fname)
    download_file(fname, dest)
    os.remove(dest)


if __name__ == "__main__":
    # Script should be run from case root directory
    if os.path.basename(os.getcwd()) == "scripts":
        print("Changing working directory to case root directory")
        os.chdir("../")

    # Create list of files on Figshare
    remote_files = get_uploaded_filenames()

    # Create list of local paths
    local_fpaths = [remote_fname_to_local(f) for f in remote_files]

    # If filename(s) are passed to script, download those
    if len(sys.argv) > 1:
        remote_files = sys.argv[1:]

    for remote, local in zip(remote_files, local_fpaths):
        if not os.path.isfile(local):
            print("Downloading {}".format(local))
            download_file(remote, local)
        else:
            print("{} already exists".format(f))
