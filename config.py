import os

basedir = os.path.abspath(os.path.dirname(__file__))
current_directory_path = os.getcwd()


class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or "19951122V@n"

    UPLOAD_FOLDER = current_directory_path + '\\users'
    MODAL_FOLDER = current_directory_path + '\\modals'


