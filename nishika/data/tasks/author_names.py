from logging import getLogger

import pandas as pd
from nishika.utils.base import Nishika

logger = getLogger(__name__)


class DownloadAozoraData(Nishika):
    """青空文庫の作品一覧データをダウンロードする"""
    def run(self):
        logger.info('download aozora author data')
        df = pd.read_csv('https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip')
        self.dump(df)


class AuthorNameData(Nishika):
    """青空文庫の著者"""
    def requires(self):
        return DownloadAozoraData()

    def run(self):
        df = self.load()
        df['name'] = df['姓'] + df['名']
        self.dump(list(df['name'].unique()))


class AkutagawaWorksData(Nishika):
    """芥川竜之介の作品"""
    def requires(self):
        return DownloadAozoraData()

    def run(self):
        df = self.load()
        df['name'] = df['姓'] + df['名']
        self.dump(list(df[df['name'] == '芥川竜之介']['作品名'].unique()))


class AozoraWorksData(Nishika):
    """芥川竜之介でない作品"""
    def requires(self):
        return DownloadAozoraData()

    def run(self):
        df = self.load()
        df['name'] = df['姓'] + df['名']
        self.dump(list(df[df['name'] != '芥川竜之介']['作品名'].unique()))
