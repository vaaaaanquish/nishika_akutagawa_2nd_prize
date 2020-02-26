from logging import getLogger
import os
import zipfile

import requests
from nishika.utils.base import Nishika
from nishika.utils.get_path import get_auxiliary_data_path

logger = getLogger(__name__)


class BertModel(Nishika):
    def run(self):
        path = get_auxiliary_data_path('Japanese_L-12_H-768_A-12_E-30_BPE')
        if not os.path.exists(path):
            self._download_jap_bert(path)
        self.dump(get_auxiliary_data_path('Japanese_L-12_H-768_A-12_E-30_BPE'))

    def _download_jap_bert(self, path):
        logger.info('make model file...')
        url = 'http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JapaneseBertPretrainedModel/Japanese_L-12' + \
            '_H-768_A-12_E-30_BPE.zip&name=Japanese_L-12_H-768_A-12_E-30_BPE.zip'
        r = requests.get(url, stream=True)
        with open(path + '.zip', 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        logger.info('extract model file.')
        zfile = zipfile.ZipFile(path + '.zip')
        zfile.extractall('/'.join(path.split('/')[:-1]))
