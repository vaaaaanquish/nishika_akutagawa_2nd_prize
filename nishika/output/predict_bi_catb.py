from logging import getLogger

import pandas as pd
from luigi.util import inherits
from nishika.model.train_bi_catb import TrainCATB
from nishika.data.load_data import LoadTestData
from nishika.model.feature_selection import MakeTestSelectionFeatureData
from nishika.utils.base import Nishika

logger = getLogger(__name__)


@inherits(TrainCATB)
@inherits(MakeTestSelectionFeatureData)
class MakeOutputCATBSubmission(Nishika):
    def requires(self):
        return dict(model=self.clone(TrainCATB), testf=self.clone(MakeTestSelectionFeatureData), testdf=LoadTestData())

    def run(self):
        model = self.load('model')
        test_df = self.load('testdf')
        test_feature = self.load('testf')

        y = model.predict(test_feature)
        df = pd.concat([test_df[['writing_id', 'times_akutagawa', 'times_flag']], pd.DataFrame(y, columns=['author'])], axis=1)
        df['author'] = df.apply(lambda x: 0 if x['times_flag'] == 1 and x['times_akutagawa'] == 0 else x['author'], axis=1)
        self.dump(df[['writing_id', 'author']])
