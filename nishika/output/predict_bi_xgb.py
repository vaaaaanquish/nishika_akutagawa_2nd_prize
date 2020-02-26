from logging import getLogger

import pandas as pd
from luigi.util import inherits
from nishika.model.feature_selection import MakeTestSelectionFeatureData, MakeTestFeatureData
from nishika.data.load_data import LoadTestData
from nishika.model.train_bi_xgb import TrainXGB, TrainNonSelectionXGB
from nishika.utils.base import Nishika
import swifter  # noqa

logger = getLogger(__name__)


@inherits(TrainXGB)
@inherits(MakeTestSelectionFeatureData)
class MakeOutputXGBSubmission(Nishika):
    def requires(self):
        return dict(model=self.clone(TrainXGB), testf=self.clone(MakeTestSelectionFeatureData), testdf=LoadTestData())

    def run(self):
        model = self.load('model')
        test_df = self.load('testdf')
        test_feature = self.load('testf')

        y = model.predict(test_feature)
        df = pd.concat([test_df[['writing_id', 'times_akutagawa', 'times_flag']], pd.DataFrame(y, columns=['author'])], axis=1)
        df['author'] = df.apply(lambda x: 0 if x['times_flag'] == 1 and x['times_akutagawa'] == 0 else x['author'], axis=1)
        self.dump(df[['writing_id', 'author']])


@inherits(TrainNonSelectionXGB)
@inherits(MakeTestFeatureData)
class MakeNonSelectedOutputXGBSubmission(Nishika):
    def requires(self):
        return dict(model=self.clone(TrainNonSelectionXGB), testf=self.clone(MakeTestFeatureData), testdf=LoadTestData())

    def run(self):
        model = self.load('model')
        test_df = self.load('testdf')
        test_feature = self.load('testf')

        y = model.predict(test_feature)
        df = pd.concat([test_df[['writing_id', 'times_akutagawa', 'times_flag']], pd.DataFrame(y, columns=['author'])], axis=1)
        df['author'] = df.apply(lambda x: 0 if x['times_flag'] == 1 and x['times_akutagawa'] == 0 else x['author'], axis=1)
        self.dump(df[['writing_id', 'author']])
