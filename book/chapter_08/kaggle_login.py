import os
# os.environ["KAGGLE_USERNAME"] = "marcel.luternauer@swisscom.com"
# os.environ["KAGGLE_KEY"] = "e57b80a7e1fe1751f7562edf8e3fd4fd"

import kagglehub
kagglehub.login()

download_path = kagglehub.competition_download("dogs-vs-cats-redux-kernels-edition")