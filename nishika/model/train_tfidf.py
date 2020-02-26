from logging import getLogger

import redshells.data
import redshells.train
from nishika.data.tasks.tokenized_data import TokenizedData
from nishika.utils.base import Nishika

logger = getLogger(__name__)


class TrainTfidf(Nishika):
    def requires(self):
        text_data = TokenizedData()
        tfidf = redshells.train.TrainTfidf(tokenized_text_data_task=text_data)
        return tfidf

    def output(self):
        return self.input()
