""" Module for working with Excel XML files from OPTA in Pandas """

import tempfile
from zipfile import ZipFile
import shutil
import os
from fnmatch import fnmatch


def change_in_zip(file_name: str, name_filter: str, change: callable):
    """ Fixer for the "synchVertical" property in the Excel files

    Args:
        file_name (str): Path to the Excel file
        name_filter (str): Filter for the files to change
        change (callable): Function to change the data
    """

    tempdir = tempfile.mkdtemp()
    try:
        tempname = os.path.join(tempdir, "new.zip")
        with ZipFile(file_name, "r") as r, ZipFile(tempname, "w") as w:
            for item in r.infolist():
                data = r.read(item.filename)
                if fnmatch(item.filename, name_filter):
                    data = change(data)
                w.writestr(item, data)
        shutil.move(tempname, file_name)
    finally:
        shutil.rmtree(tempdir)
