from logging import getLogger
import os
import subprocess
import shutil

import neologdn
from tqdm import tqdm
import MeCab

logger = getLogger(__name__)
"""
テキスト分割系
"""


def _filter(x):
    return x not in [
        '\n', '、', '。', '\r', '《', '》', '」', '「', '…', '[', ']'
        '[#', '下げ'
        '[#「', ')', '(', '?」', '?', '?', '・', '!', '…。', '|', '見出し', '\u3000', '　', '／', '″', '＼'
    ]


def exec_cmd(cmd: str, require_return=True) -> str:
    logger.info(f'execute the command: {cmd}')
    out = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = out.communicate()
    if out.returncode != 0:
        raise RuntimeError(stderr.decode())

    if not require_return:
        return ''

    results = stdout.decode().split()
    if len(results) == 0:
        raise RuntimeError(f'command  "{cmd}" throw error with message "{stderr.decode()}"')
    logger.info(f'the result: {results[0]}')
    return results[0]


def get_resource_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))


def get_user_dic_path(csv_file_names, user_dic_file_name, dic_dir=None, force_reinstall=False, install_if_not_exist=True, normalize=None) -> str:
    dic_path = os.path.join(get_resource_dir(), user_dic_file_name)
    if force_reinstall:
        make_user_dic(dic_dir=dic_dir, csv_file_names=csv_file_names, user_dic_file_name=user_dic_file_name, normalize=normalize)

    if os.path.exists(dic_path):
        return dic_path

    if install_if_not_exist:
        make_user_dic(dic_dir=dic_dir, csv_file_names=csv_file_names, user_dic_file_name=user_dic_file_name, normalize=normalize)
        return get_user_dic_path(csv_file_names, user_dic_file_name, install_if_not_exist=False)

    raise RuntimeError(f"cannot find {user_dic_file_name}. please call with force_reinstall=True.")


def merge_csv_to_temp_file(csv_file_names, normalize=None):
    csv = []
    for file_name in csv_file_names:
        with open(os.path.join(get_resource_dir(), file_name), 'r') as f:
            csv += f.read().splitlines()
    if normalize is not None:

        def normalize_word(line, normalize_func):
            xs = line.split(',')
            xs[0] = normalize_func(xs[0])
            return ','.join(xs)

        csv = [normalize_word(line, normalize) for line in csv]
    output_file = os.path.join(get_resource_dir(), 'temp_merge_csv_to_temp_file.csv')
    with open(output_file, 'w') as f:
        f.write('\n'.join(csv))
    return output_file


def make_user_dic(dic_dir, csv_file_names, user_dic_file_name, normalize):
    logger.info(f"install user dic with dic_dir={dic_dir}, csv_file_names={csv_file_names}")
    mecab_dict_index = os.path.join(exec_cmd("mecab-config --libexecdir"), 'mecab-dict-index')
    dic_dir = dic_dir or os.path.dirname(exec_cmd("mecab -D | grep 'filename' | sed -e 's/filename:\t//'"))
    csv_file_path = merge_csv_to_temp_file(csv_file_names, normalize=normalize)
    dic_file_path = os.path.join(get_resource_dir(), user_dic_file_name)
    exec_cmd(f"{mecab_dict_index} -d {dic_dir} -u {dic_file_path} -f utf-8 -t utf-8 {csv_file_path}")
    os.remove(csv_file_path)
    logger.info(f'the dictionary path is {dic_file_path}')


def _install_dic():
    logger.info("install neologd")
    work_dir = 'resources/neologd/'
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    exec_cmd(f"git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git {work_dir}", require_return=False)
    exec_cmd(f"{os.path.join(work_dir, 'bin/install-mecab-ipadic-neologd')} -n -y", require_return=False)


def _get_dicdir():
    return os.path.join(exec_cmd("mecab-config --dicdir"), 'mecab-ipadic-neologd')


def get_dic_path(force_reinstall=False, install_if_not_exist=True):
    if force_reinstall:
        _install_dic()

    if os.path.exists(_get_dicdir()):
        return _get_dicdir()

    if install_if_not_exist:
        _install_dic()
        return get_dic_path(install_if_not_exist=False)
    return _get_dicdir()


def make_mecab_tagger(words=False):
    try:
        neologd_dic = get_dic_path()
        logger.info(f'neologd path: {neologd_dic}')
        if words:
            return MeCab.Tagger(f'-Ochasen --dicdir={neologd_dic}')
        return MeCab.Tagger(f'-Owakati --dicdir={neologd_dic}')
    except FileNotFoundError:
        return


def get_wf(df, wf):
    feature = []
    for x, l in tqdm(zip(df['tokenized_body'], df['text_length'])):
        n = [x.count(w) / l for w, _ in wf]
        feature.append(n)
    return feature


def get_words(tokenizer, document):
    tokenizer.parse("")
    node = tokenizer.parseToNode(document.replace(',', '、'))
    keywords = []
    while node:
        if node.feature.split(",")[0] == u"名詞":
            keywords.append(node.surface)
        elif node.feature.split(",")[0] == u"形容詞":
            keywords.append(node.feature.split(",")[6])
        elif node.feature.split(",")[0] == u"動詞":
            keywords.append(node.feature.split(",")[6])
        node = node.next
    return keywords


def tokenize_text(text, words=False):
    text = neologdn.normalize(text)
    if words:
        return [x for x in get_words(self.mecab, text) if _filter(x)]
    return [x for x in self.mecab.parse(text).split(' ') if _filter(x)]
