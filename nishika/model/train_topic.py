from logging import getLogger

import redshells.data
import redshells.train
from nishika.data.tasks.tokenized_data import TokenizedData
from nishika.utils.base import Nishika

logger = getLogger(__name__)


class TrainLDA(Nishika):
    def requires(self):
        text_data = TokenizedData()
        dictionary = redshells.train.TrainDictionary(tokenized_text_data_task=text_data)
        lda = redshells.train.TrainLdaModel(tokenized_text_data_task=text_data, dictionary_task=dictionary)
        return lda

    def output(self):
        return self.input()
