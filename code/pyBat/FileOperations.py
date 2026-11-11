import glob
import os
import shutil
import sys
import re

import shutil
import easygui


def create_folder(folder, remove_first=False, ask=False):
    contents = []
    proceed = True
    full_path = os.path.abspath(folder)
    if os.path.isdir(folder): contents = os.listdir(folder)
    message = 'The folder is not empty. Do you want to delete it?\n' + full_path
    if len(contents) > 0 and remove_first:
        if ask: proceed = easygui.ynbox(message, 'Warning', ('Yes', 'No'))
        if not proceed: return False
        shutil.rmtree(folder)
    if not os.path.isdir(folder): os.makedirs(folder)
    return True


def check_file_exists(file, quit_on_no=False, ask=False):
    if quit_on_no: ask = True
    exists = os.path.isfile(file)
    if not ask: return exists
    full_path = os.path.abspath(file)
    proceed = True
    message = 'The file exists. Do you want to continue?\n' + full_path
    if exists: proceed = easygui.ynbox(message, 'Warning', ('Yes', 'No'))
    if not proceed and quit_on_no: sys.exit()
    return exists


def get_folders(directory='', pattern='*'):
    # folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    folders = glob.glob("./%s/%s/" % (directory, pattern))
    return folders


def get_files(directory='.', pattern='*'):
    files = [f for f in os.listdir(directory) if re.match(pattern, f)]
    return files
