from logging import getLogger

from nishika.data.tasks.load_data import LoadTestData, LoadTrainData
from nishika.data.functions.tokenize import tokenize_text, make_mecab_tagger
from nishika.utils.base import MecabBase
from tqdm import tqdm

logger = getLogger(__name__)


class TokenizedData(MecabBase):
    """文書データのトークナイズ"""
    def requires(self):
        return dict(test=LoadTestData(), train=LoadTrainData())

    def output(self):
        return self.make_target('data/TokenizedData.pkl', use_unique_id=False)

    def run(self):
        test_df = self.load('test')
        train_df = self.load('train')
        mecab = make_mecab_tagger(self.words)
        logger.info('made mecab tagger')
        text_list = train_df['body'].values.tolist() + test_df['body'].values.tolist()
        text_list = [tokenize_text(mecab, x, self.words) for x in tqdm(text_list)]
        self.dump(text_list)
