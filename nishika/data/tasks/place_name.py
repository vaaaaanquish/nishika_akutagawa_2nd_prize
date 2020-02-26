from logging import getLogger
import re
import os
import zipfile

import requests
import pandas as pd
from nishika.utils.get_path import get_auxiliary_data_path
from nishika.utils.base import Nishika
import regex

logger = getLogger(__name__)

ap = re.compile(r'[a-zA-Z\s]+')
ktp = regex.compile(r'\p{Katakana}+')


class DownloadPlaceData(Nishika):
    """地名データのダウンロードと解凍"""
    rerun = True

    def run(self):
        zip_file_path = get_auxiliary_data_path('gazetteer.zip')
        if not os.path.exists(zip_file_path):
            logger.info('get place file...')
            r = requests.get('https://int.nihu.jp/resource/data/gazetteer.zip', stream=True)
            with open(zip_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()

        dir_path = get_auxiliary_data_path('gazetteer')
        if not os.path.exists(dir_path):
            logger.info('extract file.')
            with zipfile.ZipFile(zip_file_path) as z:
                for info in z.infolist():
                    info.filename = info.filename.encode('cp437').decode('cp932')
                    z.extract(info, path=get_auxiliary_data_path('gazetteer'))

        self.dump(['done'])


class MakePlaceData(Nishika):
    """地名データの整形"""
    def requires(self):
        return DownloadPlaceData()

    def run(self):
        _ = self.load()
        # データ整形
        df = pd.read_csv(get_auxiliary_data_path('gazetteer/地名.txt'))
        zokusei_df = pd.read_csv(get_auxiliary_data_path('gazetteer/地名_属性.txt'))
        df = pd.merge(df, zokusei_df[['番号', '属性']].rename(columns={'属性': 'attribute', '番号': '属性'}), on='属性', how='left')

        joui_df = pd.read_csv(get_auxiliary_data_path('gazetteer/地名_上位地名.txt'))
        df = pd.merge(df[['ID', '地名', 'attribute']],
                      joui_df[['ID', '地名ID', '上位地名ID', '上位地名']].rename(columns={
                          'ID': 'joui_id',
                          '地名ID': 'ID'
                      }),
                      on='ID',
                      how='left')

        betsumei_df = pd.read_csv(get_auxiliary_data_path('gazetteer/地名_別名.txt'))
        df = pd.merge(df, betsumei_df[['地名ID', '別名']].rename(columns={'地名ID': 'ID'}), on='ID', how='left')

        df = df.rename(columns={'地名': 'name', '上位地名': 'top_name', '別名': 'other_name'})[['name', 'top_name', 'other_name', 'attribute']]

        # 都道府県
        top_set = set()
        for x in df['top_name'].unique():
            if type(x) != str:
                continue
            top_set.add(x)
            top_set.add(x.rstrip('郡'))
            top_set.add(x.rstrip('県'))
            top_set.add(x.rstrip('国'))
            top_set.add(x.rstrip('区'))
            top_set.add(x.rstrip('府'))
            top_set.add(x.rstrip('都'))
            top_set.add(x.rstrip('（北'))

        # 地名
        name_attr_dict = df[['attribute', 'name']].groupby('attribute')['name'].apply(set)

        # 地名別名
        tmp = df[['other_name', 'attribute']]
        tmp = tmp[tmp['other_name'].apply(self.if_kanji)]
        other_attr_dict = tmp[['attribute', 'other_name']].groupby('attribute')['other_name'].apply(set)

        # merge
        attr_dict = {}
        for i, (k, v) in enumerate(name_attr_dict.items()):
            attr_dict[f'attr_{i}'] = v | other_attr_dict.get(k, set())

        attr_dict[f'attr_{i+1}'] = top_set

        self.dump(attr_dict)

    def if_kanji(self, x):
        if type(x) != str:
            return False
        if ap.match(x) is None and ktp.match(x) is None:
            return True
        return False
