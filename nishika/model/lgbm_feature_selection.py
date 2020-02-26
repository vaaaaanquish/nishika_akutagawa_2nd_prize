from logging import getLogger
import luigi
import redshells
import pandas as pd
from luigi.util import inherits
from nishika.data.tasks.make_feature_data import MakeTrainFeatureData, MakeTestFeatureData
from nishika.utils.base import Nishika

logger = getLogger(__name__)


@inherits(MakeTrainFeatureData)
class TrainOptimizeLGBM(Nishika):
    def requires(self):
        data = self.clone(MakeTrainFeatureData)
        return redshells.train.OptimizeBinaryClassificationModel(rerun=True,
                                                                 train_data_task=data,
                                                                 target_column_name='author',
                                                                 model_name='LGBMClassifier',
                                                                 test_size=0.2,
                                                                 optuna_param_name='LGBMClassifier_default')

    def run(self):
        model = self.load()
        self.dump(model)


@inherits(MakeTrainFeatureData)
@inherits(TrainOptimizeLGBM)
class TrainLGBM(Nishika):
    def requires(self):
        data = self.clone(MakeTrainFeatureData)
        task = self.clone(TrainOptimizeLGBM)
        if not task.complete():
            luigi.build([task], local_scheduler=True)
        param = task.output().load()
        return redshells.train.TrainClassificationModel(rerun=True,
                                                        train_data_task=data,
                                                        target_column_name='author',
                                                        model_name='LGBMClassifier',
                                                        model_kwargs=param['best_params'])

    def run(self):
        model = self.load()
        self.dump(model)


@inherits(TrainLGBM)
class LGBMFeatureSelection(Nishika):
    def requires(self):
        return self.clone(TrainLGBM)

    def run(self):
        model = self.load()
        self.dump([i for i, x in enumerate(model.feature_importances_) if x > 0])


@inherits(MakeTrainFeatureData)
@inherits(LGBMFeatureSelection)
class MakeTrainSelectionFeatureData(Nishika):
    def requires(self):
        return dict(importance=self.clone(LGBMFeatureSelection), data=self.clone(MakeTrainFeatureData))

    def run(self):
        importance = self.load('importance')
        data = self.load('data')
        author = data[['author']].copy()
        data = data.drop(columns='author').iloc[:, importance]
        data = pd.concat([data, author], axis=1)
        self.dump(data)


@inherits(MakeTestFeatureData)
@inherits(LGBMFeatureSelection)
class MakeTestSelectionFeatureData(Nishika):
    def requires(self):
        return dict(importance=self.clone(LGBMFeatureSelection), data=self.clone(MakeTestFeatureData))

    def run(self):
        importance = self.load('importance')
        data = self.load('data')
        select_data = []
        for x in data:
            select_data.append([x[i] for i in importance])
        self.dump(select_data)
