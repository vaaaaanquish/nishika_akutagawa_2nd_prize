from logging import getLogger

import luigi
from nishika.data.tasks.tokenized_data import TokenizedData
from nishika.utils.base import Nishika
from collections import Counter

logger = getLogger(__name__)


class WordFeature(Nishika):
    """単語のカウント特徴の生成"""
    most_common = luigi.IntParameter()

    def requires(self):
        return TokenizedData()

    def run(self):
        data = self.load()
        c = Counter()
        for words in data:
            c += Counter(words)
        self.dump(c.most_common(self.most_common))
