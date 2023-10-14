import zipfile


def unzip(src, dst):
    if zipfile.is_zipfile(src):
        f = zipfile.ZipFile(src, 'r')
        for file in f.namelist():
            f.extract(file, dst)
    else:
        raise FileNotFoundError