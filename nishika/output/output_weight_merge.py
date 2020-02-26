from logging import getLogger

from luigi.util import inherits
from nishika.output.predict_bi_xgb import MakeOutputXGBSubmission, MakeNonSelectedOutputXGBSubmission
from nishika.output.predict_bi_lgbm import MakeOutputLGBMSubmission, MakeNonSelectedOutputLGBMSubmission
from nishika.output.predict_bi_catb import MakeOutputCATBSubmission
from nishika.output.predict_bi_nn import MakeOutputNNSubmission
from nishika.utils.base import Nishika
import pandas as pd

logger = getLogger(__name__)


@inherits(MakeOutputCATBSubmission)
@inherits(MakeOutputLGBMSubmission)
@inherits(MakeOutputXGBSubmission)
@inherits(MakeOutputNNSubmission)
@inherits(MakeNonSelectedOutputLGBMSubmission)
@inherits(MakeNonSelectedOutputXGBSubmission)
class MergeFiles(Nishika):
    def requires(self):
        return dict(xgb=self.clone(MakeOutputXGBSubmission),
                    lgbm=self.clone(MakeOutputLGBMSubmission),
                    catb=self.clone(MakeOutputCATBSubmission),
                    nn=self.clone(MakeOutputNNSubmission),
                    lgbmn=self.clone(MakeNonSelectedOutputLGBMSubmission),
                    xgbn=self.clone(MakeNonSelectedOutputXGBSubmission),
                    xgbw=self.clone(MakeOutputXGBSubmission, words=True),
                    lgbmw=self.clone(MakeOutputLGBMSubmission, words=True),
                    catbw=self.clone(MakeOutputCATBSubmission, words=True),
                    nnw=self.clone(MakeOutputNNSubmission, words=True),
                    xgbnw=self.clone(MakeNonSelectedOutputXGBSubmission, words=True),
                    lgbmnw=self.clone(MakeNonSelectedOutputLGBMSubmission, words=True))

    def run(self):
        data = ['xgb', 'lgbm', 'catb', 'nn', 'lgbmn', 'xgbn', 'xgbw', 'lgbmw', 'catbw', 'nnw', 'lgbmnw', 'xgbnw']

        df = self.load(data[0])
        df = df.rename(columns={'author': 'col_' + data[0]})
        for x in data[1:]:
            tmp = self.load(x)
            tmp = tmp.rename(columns={'author': f'col_{x}'})
            df = pd.merge(df, tmp, on='writing_id')
        self.dump(df)


class WeightEnsemble(Nishika):
    def requires(self):
        return MergeFiles()

    def output(self):
        return self.make_target(f'output/ensemble_submit.csv')

    def run(self):
        df = self.load()
        df['author'] = df.apply(self.ensemble, axis=1)
        self.dump(df[['writing_id', 'author']])

    def ensemble(self, df):
        data = ['xgb', 'lgbm', 'catb', 'nn', 'xgbw', 'lgbmw', 'catbw', 'nnw']
        weight = 0
        for x in data:
            if x == 'catbw':
                weight += df[f'col_{x}'] * 2
            else:
                weight += df[f'col_{x}']
        return 1 if weight > 1 else 0
