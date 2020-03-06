from logging import getLogger

import pandas as pd
import numpy as np
from nishika.data.tasks.load_data import LoadTrainData, LoadTestData
from nishika.data.tasks.word_feature import WordFeature
from nishika.data.functions.add_feature import add_nlp_feature
from nishika.model.train_scdv import TrainSCDV
from nishika.model.train_tfidf import TrainTfidf
from nishika.model.train_lda import TrainLDA
from nishika.model.bert import BertModel
from nishika.utils.base import FeatuerBase

logger = getLogger(__name__)


class MakeTrainFeatureData(FeatuerBase):
    """predict用モデル学習のための特徴量DF生成"""
    def requires(self):
        return dict(scdv=TrainSCDV(), tfidf=TrainTfidf(), train=LoadTrainData(), lda=TrainLDA(), wf=WordFeature(), bert=BertModel())

    def run(self):
        train_df = self.load('train')
        scdv_f, tf_f, lda_f, data_f, tft_f, bert_f, word_f = add_nlp_feature(train_df, self.load('scdv'), self.load('tfidf'), self.load('lda'), self.load('wf'),
                                                                             self.tfidf_top_k, self.train_cols, self.load('bert'), self.words)

        logger.info('features concat')
        train_df = pd.concat([
            train_df['author'],
            pd.DataFrame(scdv_f),
            pd.DataFrame(tf_f),
            pd.DataFrame(lda_f),
            pd.DataFrame(data_f),
            pd.DataFrame(tft_f),
            pd.DataFrame(bert_f),
            pd.DataFrame(word_f)
        ],
                             axis=1)
        self.dump(train_df)


class MakeTestFeatureData(FeatuerBase):
    """predictのためのTest用の配列生成"""
    def requires(self):
        return dict(scdv=TrainSCDV(), test=LoadTestData(), tfidf=TrainTfidf(), lda=TrainLDA(), wf=WordFeature(), bert=BertModel())

    def run(self):
        scdv_f, tf_f, lda_f, data_f, tft_f, bert_f, word_f = add_nlp_feature(self.load('test'), self.load('scdv'), self.load('tfidf'), self.load('lda'),
                                                                             self.load('wf'), self.tfidf_top_k, self.train_cols, self.load('bert'), self.words)

        self.dump([
            np.append(np.append(np.append(np.append(np.append(np.append(x, y), z), w), k), b), f)
            for x, y, z, w, k, b, f in zip(scdv_f, tf_f, lda_f, data_f, tft_f, bert_f, word_f)
        ])
