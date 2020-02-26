import luigi
import gokart


class Nishika(gokart.TaskOnKart):
    task_namespace = 'nishika'


class MecabBase(Nishika):
    words = luigi.BoolParameter()


class FeatuerBase(MecabBase):
    tfidf_top_k = luigi.IntParameter()
    train_cols = luigi.ListParameter()
