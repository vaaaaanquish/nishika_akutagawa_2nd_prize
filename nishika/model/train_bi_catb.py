from logging import getLogger

import luigi
import redshells
from luigi.util import inherits
from nishika.model.lgbm_feature_selection import MakeTrainSelectionFeatureData
from nishika.utils.base import Nishika

logger = getLogger(__name__)


@inherits(MakeTrainSelectionFeatureData)
class TrainOptimizeCATB(Nishika):
    def requires(self):
        data = self.clone(MakeTrainSelectionFeatureData)
        return redshells.train.OptimizeBinaryClassificationModel(rerun=True,
                                                                 train_data_task=data,
                                                                 target_column_name='author',
                                                                 model_name='CatBoostClassifier',
                                                                 test_size=0.2,
                                                                 optuna_param_name='CatBoostClassifier_default')

    def run(self):
        model = self.load()
        self.dump(model)


@inherits(MakeTrainSelectionFeatureData)
@inherits(TrainOptimizeCATB)
class TrainCATB(Nishika):
    def requires(self):
        data = self.clone(MakeTrainSelectionFeatureData)
        task = self.clone(TrainOptimizeCATB)
        if not task.complete():
            luigi.build([task], local_scheduler=True)
        param = task.output().load()
        return redshells.train.TrainClassificationModel(rerun=True,
                                                        train_data_task=data,
                                                        target_column_name='author',
                                                        model_name='CatBoostClassifier',
                                                        model_kwargs=param['best_params'])

    def run(self):
        model = self.load()
        self.dump(model)
