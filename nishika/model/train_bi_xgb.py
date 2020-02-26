from logging import getLogger

import redshells
import gokart
from luigi.util import inherits
from nishika.model.feature_selection import MakeTrainSelectionFeatureData, MakeTrainFeatureData
from nishika.utils.base import Nishika
import swifter  # noqa

logger = getLogger(__name__)


@inherits(MakeTrainSelectionFeatureData)
class TrainOptimizeXGB(Nishika):
    def requires(self):
        data = self.clone(MakeTrainSelectionFeatureData)
        return redshells.train.OptimizeBinaryClassificationModel(rerun=True,
                                                                 train_data_task=data,
                                                                 target_column_name='author',
                                                                 model_name='XGBClassifier',
                                                                 model_kwargs=dict(n_estimators=50),
                                                                 test_size=0.2,
                                                                 optuna_param_name='XGBClassifier_default')

    def run(self):
        model = self.load()
        self.dump(model)


@inherits(MakeTrainSelectionFeatureData)
@inherits(TrainOptimizeXGB)
class TrainXGB(Nishika):
    def requires(self):
        data = self.clone(MakeTrainSelectionFeatureData)
        task = self.clone(TrainOptimizeXGB)
        if not task.complete():
            gokart.build([task], local_scheduler=True)
        param = task.output().load()
        return redshells.train.TrainClassificationModel(rerun=True,
                                                        train_data_task=data,
                                                        target_column_name='author',
                                                        model_name='XGBClassifier',
                                                        model_kwargs=param['best_params'])

    def run(self):
        model = self.load()
        self.dump(model)


@inherits(MakeTrainFeatureData)
class TrainNonSelectionOptimizeXGB(Nishika):
    def requires(self):
        data = self.clone(MakeTrainFeatureData)
        return redshells.train.OptimizeBinaryClassificationModel(rerun=True,
                                                                 train_data_task=data,
                                                                 target_column_name='author',
                                                                 model_name='XGBClassifier',
                                                                 model_kwargs=dict(n_estimators=50),
                                                                 test_size=0.2,
                                                                 optuna_param_name='XGBClassifier_default')

    def run(self):
        model = self.load()
        self.dump(model)


@inherits(MakeTrainFeatureData)
@inherits(TrainNonSelectionOptimizeXGB)
class TrainNonSelectionXGB(Nishika):
    def requires(self):
        data = self.clone(MakeTrainFeatureData)
        task = self.clone(TrainNonSelectionOptimizeXGB)
        if not task.complete():
            gokart.build([task], local_scheduler=True)
        param = task.output().load()
        return redshells.train.TrainClassificationModel(rerun=True,
                                                        train_data_task=data,
                                                        target_column_name='author',
                                                        model_name='XGBClassifier',
                                                        model_kwargs=param['best_params'])

    def run(self):
        model = self.load()
        self.dump(model)
