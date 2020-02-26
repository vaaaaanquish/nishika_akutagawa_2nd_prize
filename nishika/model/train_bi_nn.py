from logging import getLogger

import numpy as np
from luigi.util import inherits
from skorch.classifier import NeuralNetBinaryClassifier
from nishika.model.lgbm_feature_selection import MakeTrainSelectionFeatureData, MakeTestSelectionFeatureData
from nishika.utils.base import Nishika
from torch import nn
import torch.nn.functional as F
from skorch.dataset import CVSplit
from sklearn.preprocessing import StandardScaler

logger = getLogger(__name__)

INPUT_LEN = 0


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        global INPUT_LEN
        self.layer1 = nn.Linear(INPUT_LEN, 256)
        self.layer2 = nn.Linear(256, 10, bias=True)
        self.layer3 = nn.Linear(10, 1, bias=True)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


@inherits(MakeTrainSelectionFeatureData)
@inherits(MakeTestSelectionFeatureData)
class MakeStandardScalar(Nishika):
    def requires(self):
        return {'train': self.clone(MakeTrainSelectionFeatureData), 'test': self.clone(MakeTestSelectionFeatureData)}

    def run(self):
        train = self.load('train')
        test = self.load('test')
        train = train.drop(columns=['author']).values
        data = np.concatenate([train, np.array(test)], axis=0)
        sc = StandardScaler()
        sc.fit(data)
        self.dump(sc)


@inherits(MakeTrainSelectionFeatureData)
@inherits(MakeStandardScalar)
class TrainNN(Nishika):
    def requires(self):
        return {'data': self.clone(MakeTrainSelectionFeatureData), 'sc': self.clone(MakeStandardScalar)}

    def run(self):
        data = self.load('data')
        sc = self.load('sc')
        Y = data['author'].values
        Y = Y.astype('float32')
        X = data.drop(columns=['author']).values
        X = sc.transform(X)
        global INPUT_LEN
        INPUT_LEN = len(X[0])
        model = NeuralNetBinaryClassifier(NN, max_epochs=100, iterator_train__shuffle=True, train_split=CVSplit(cv=10, stratified=True, random_state=0))
        model.fit(X, Y)
        self.dump(model)
