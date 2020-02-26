from logging import getLogger

import pandas as pd
from nishika.utils.base import Nishika
from nishika.data.functions.add_feature import add_rule_feature, add_place_feature
from nishika.data.tasks.place_name import MakePlaceData
from nishika.data.tasks.author_names import AuthorNameData, AkutagawaWorksData, AozoraWorksData

logger = getLogger(__name__)


class LoadTrainData(Nishika):
    """Trainデータ生成"""
    def requires(self):
        return dict(place=MakePlaceData(), author=AuthorNameData(), akutagawa=AkutagawaWorksData(), aozora=AozoraWorksData())

    def run(self):
        place_dict = self.load('place')
        author_names = self.load('author')
        akutagawa_works = self.load('akutagawa')
        aozora_works = self.load('aozora')
        df = pd.read_csv('./data/train.csv')

        df = add_place_feature(df, place_dict)
        df = add_rule_feature(df, author_names, akutagawa_works, aozora_works)
        self.dump(df)


class LoadTestData(Nishika):
    """Testデータ生成"""
    def requires(self):
        return dict(place=MakePlaceData(), author=AuthorNameData(), akutagawa=AkutagawaWorksData(), aozora=AozoraWorksData())

    def run(self):
        place_dict = self.load('place')
        author_names = self.load('author')
        akutagawa_works = self.load('akutagawa')
        aozora_works = self.load('aozora')
        df = pd.read_csv('./data/test.csv')

        df = add_place_feature(df, place_dict)
        df = add_rule_feature(df, author_names, akutagawa_works, aozora_works)
        self.dump(df)
