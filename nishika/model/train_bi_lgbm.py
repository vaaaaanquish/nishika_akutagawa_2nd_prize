from logging import getLogger

import gokart
import redshells
from luigi.util import inherits
from nishika.model.feature_selection import MakeTrainSelectionFeatureData
from nishika.utils.base import Nishika

logger = getLogger(__name__)


@inherits(MakeTrainSelectionFeatureData)
class TrainSelectedOptimizeLGBM(Nishika):
    def requires(self):
        data = self.clone(MakeTrainSelectionFeatureData)
        return redshells.train.OptimizeBinaryClassificationModel(rerun=True,
                                                                 train_data_task=data,
                                                                 target_column_name='author',
                                                                 model_name='LGBMClassifier',
                                                                 test_size=0.2,
                                                                 optuna_param_name='LGBMClassifier_default')

    def run(self):
        model = self.load()
        self.dump(model)


@inherits(MakeTrainSelectionFeatureData)
@inherits(TrainSelectedOptimizeLGBM)
class TrainSelectedLGBM(Nishika):
    def requires(self):
        data = self.clone(MakeTrainSelectionFeatureData)
        task = self.clone(TrainSelectedOptimizeLGBM)
        if not task.complete():
            gokart.run([task], local_scheduler=True)
        param = task.output().load()
        return redshells.train.TrainClassificationModel(rerun=True,
                                                        train_data_task=data,
                                                        target_column_name='author',
                                                        model_name='LGBMClassifier',
                                                        model_kwargs=param['best_params'])

    def run(self):
        model = self.load()
        self.dump(model)
