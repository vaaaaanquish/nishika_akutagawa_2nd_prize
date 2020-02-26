from logging import getLogger

import luigi
import redshells.data
import redshells.train
from nishika.data.tasks.tokenized_data import TokenizedData
from nishika.utils.base import Nishika

logger = getLogger(__name__)


class TrainSCDV(Nishika):
    cluster_size = luigi.IntParameter()
    text_sample_size = luigi.IntParameter()

    def requires(self):
        text_data = TokenizedData()
        dictionary = redshells.train.TrainDictionary(tokenized_text_data_task=text_data)
        fasttext = redshells.train.TrainFastText(tokenized_text_data_task=text_data)
        scdv = redshells.train.TrainSCDV(tokenized_text_data_task=text_data,
                                         cluster_size=self.cluster_size,
                                         dictionary_task=dictionary,
                                         word2vec_task=fasttext,
                                         text_sample_size=self.text_sample_size)
        return scdv

    def output(self):
        return self.input()
