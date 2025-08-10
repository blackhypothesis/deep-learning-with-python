import zipfile
with zipfile.ZipFile(download_path + "/train.zip", "r") as zip_ref:
    zip_ref.extractall("data")