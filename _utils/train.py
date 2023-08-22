import os, shutil

def backupPreviousModel(dirs):
    if os.path.exists(dirs["model_path"]):
        shutil.copy(dirs["model_path"], dirs["model_path"][:-4] + '_backup.pkl')
    if os.path.exists(dirs["config_path"]):
        shutil.copy(dirs["config_path"], dirs["config_path"][:-5] + '_backup.json')