from logging import getLogger

import pandas as pd
from luigi.util import inherits
from nishika.model.train_bi_nn import TrainNN, MakeStandardScalar
from nishika.data.load_data import LoadTestData
from nishika.model.feature_selection import MakeTestSelectionFeatureData
from nishika.utils.base import Nishika
import numpy as np

logger = getLogger(__name__)


@inherits(TrainNN)
@inherits(MakeTestSelectionFeatureData)
@inherits(MakeStandardScalar)
class MakeOutputNNSubmission(Nishika):
    def requires(self):
        return dict(model=self.clone(TrainNN), testf=self.clone(MakeTestSelectionFeatureData), testdf=LoadTestData(), sc=self.clone(MakeStandardScalar))

    def run(self):
        model = self.load('model')
        test_df = self.load('testdf')
        test_feature = self.load('testf')
        sc = self.load('sc')

        test_feature = np.array(test_feature)
        test_feature = test_feature.astype('float64')
        test_feature = sc.transform(test_feature)
        y = model.predict(test_feature)
        df = pd.concat([test_df[['writing_id', 'times_akutagawa', 'times_flag']], pd.DataFrame(y, columns=['author'])], axis=1)
        df['author'] = df.apply(lambda x: 0 if x['times_flag'] == 1 and x['times_akutagawa'] == 0 else x['author'], axis=1)
        self.dump(df[['writing_id', 'author']])
