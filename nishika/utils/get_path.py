import os


def get_auxiliary_data_path(file_name: str):
    """絶対パス取得するやつ"""
    d = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'auxiliary'))
    if not os.path.exists(d):
        os.mkdir(d)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'auxiliary', file_name))
