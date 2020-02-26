from logging import getLogger
from datetime import date
from nishika.data.functions.re_functions import *
from nishika.data.functions.get_time import get_times, get_date
from nishika.data.functions.tokenized_data import make_mecab_tagger, get_words, tokenize_text, get_wf
from nishika.utils.bert_juman import BertWithJumanModel
import numpy as np
import swifter  # noqa

logger = getLogger(__name__)
"""
学習済みNLPモデルによる特徴量追加
"""


def add_nlp_feature(df, scdv_model, tf_model, lda_model, word_feature, tfidf_top_k, train_cols, bert_path, words_flag=False):
    bert_model = BertWithJumanModel(bert_path)
    mecab = make_mecab_tagger(words_flag)
    if words_flag:
        df['tokenized_body'] = df['body'].swifter.apply(lambda x: get_words(mecab, x))
    else:
        df['tokenized_body'] = df['body'].swifter.apply(lambda x: tokenize_text(mecab, x))

    logger.info('make scdv feature...')
    scdv_f = scdv_model.infer_vector(df['tokenized_body'].values.tolist())

    logger.info('make tfidf feature...')
    tf = tf_model.apply(df['tokenized_body'].values.tolist())
    ff = [[w[0] for w in sorted(x, key=lambda t: t[1], reverse=True)][:tfidf_top_k] for x in tf]
    tf_f = scdv_model.infer_vector(ff)
    tft_f = lda_model.get_document_topics(ff)

    logger.info('make lda feature...')
    lda_f = lda_model.get_document_topics(df['tokenized_body'].values.tolist())

    logger.info('make data feature...')
    data_f = [np.array([x[c] for c in train_cols]) for _, x in df.iterrows()]

    logger.info('make word feature...')
    word_f = get_wf(df[['tokenized_body', 'text_length']], word_feature)

    logger.info('make bert feature...')
    bert_f = [bert_model.get_sentence_embedding(x, pooling_layer=-2, pooling_strategy='REDUCE_MEAN_MAX') for x in df['body']]

    return scdv_f, tf_f, lda_f, data_f, tft_f, bert_f, word_f


"""
ルールベースの特徴量追加
"""


def get_place_count(x, place):
    """地名カウンタ"""
    return sum([x.count(p) if type(x) == str else 0 for p in place])


def add_place_feature(df, place_dict):
    """地名特徴量生成"""
    length = len(place_dict)
    for i in range(length):
        logger.info(f'make place {i}/{length}')
        df[f'place_{i}'] = df['body'].swifter.apply(get_place_count, place=place_dict[f'attr_{i}'])
        df[f'place_{i}_per'] = df[f'place_{i}'] / df['body'].str.len()
    return df


def add_rule_feature(df, author_names, akutagawa_works, aozora_works):
    """全てのルールベース特徴量の追加"""
    logger.info(f'make data feature')
    df['rubi_num'] = df['body'].swifter.apply(get_rubi_num)
    df['line_num'] = df['body'].swifter.apply(get_line_num)
    df['lines_num'] = df['body'].swifter.apply(get_lines_num)
    df['lines_num'] = df['body'].swifter.apply(get_lines_num)
    df['top_kuhaku'] = df['body'].swifter.apply(top_kuhaku)
    df['zenkuhaku_num'] = df['body'].swifter.apply(get_zenkaku_kuhaku_num)
    df['tenkai_num'] = df['body'].swifter.apply(get_tenkai_num)
    df['tenkai_per'] = df['tenkai_num'] / df['body'].str.len()
    df['kukai_num'] = df['body'].swifter.apply(get_kukai_num)
    df['kukai_per'] = df['kukai_num'] / df['body'].str.len()
    df['kurik_num'] = df['body'].swifter.apply(get_kurik_num)
    df['kurik_per'] = df['kurik_num'] / df['body'].str.len()
    df['suijun_num'] = df['body'].swifter.apply(get_suijun_num)
    df['suijun_per'] = df['suijun_num'] / df['body'].str.len()
    df['heading_num'] = df['body'].swifter.apply(get_heading_num)
    df['heading_per'] = df['heading_num'] / df['body'].str.len()
    df['zisage_num'] = df['body'].swifter.apply(lambda x: x.count('字下げ') if type(x) == str else 0)
    df['zisage_per'] = df['zisage_num'] / df['body'].str.len()
    df['midashi_num'] = df['body'].swifter.apply(lambda x: x.count('中見出し') if type(x) == str else 0)
    df['midashi_per'] = df['midashi_num'] / df['body'].str.len()
    df['tate_num'] = df['body'].swifter.apply(lambda x: x.count('縦中横') if type(x) == str else 0)
    df['tate_per'] = df['tate_num'] / df['body'].str.len()
    df['zizume_num'] = df['body'].swifter.apply(lambda x: x.count('字詰め') if type(x) == str else 0)
    df['zizume_per'] = df['zizume_num'] / df['body'].str.len()
    df['keisen_num'] = df['body'].swifter.apply(lambda x: x.count('罫囲み') if type(x) == str else 0)
    df['keisen_per'] = df['keisen_num'] / df['body'].str.len()
    df['kantanhu_num'] = df['body'].swifter.apply(lambda x: x.count('感嘆符') if type(x) == str else 0)
    df['kantanhu_per'] = df['kantanhu_num'] / df['body'].str.len()
    df['bouten_num'] = df['body'].swifter.apply(lambda x: x.count('傍点') if type(x) == str else 0)
    df['bouten_per'] = df['bouten_num'] / df['body'].str.len()
    df['cyuou_num'] = df['body'].swifter.apply(lambda x: x.count('左右中央') if type(x) == str else 0)
    df['cyuou_per'] = df['cyuou_num'] / df['body'].str.len()
    df['zicyu_num'] = df['body'].swifter.apply(lambda x: x.count('自注') if type(x) == str else 0)
    df['zicyu_per'] = df['zicyu_num'] / df['body'].str.len()
    df['chimozi_num'] = df['body'].swifter.apply(lambda x: x.count('小さな文字') if type(x) == str else 0)
    df['chimozi_per'] = df['chimozi_num'] / df['body'].str.len()
    df['hutozi_num'] = df['body'].swifter.apply(lambda x: x.count('太字') if type(x) == str else 0)
    df['hutozi_per'] = df['hutozi_num'] / df['body'].str.len()
    df['romanum_num'] = df['body'].swifter.apply(lambda x: x.count('ローマ数字') if type(x) == str else 0)
    df['romanum_per'] = df['romanum_num'] / df['body'].str.len()
    df['kuhaku_num'] = df['body'].swifter.apply(get_kuhaku_num)
    df['kuhaku_per'] = df['kuhaku_num'] / df['body'].str.len()
    df['bun_num'] = df['body'].swifter.apply(get_bun_num)
    df['bun10_num'] = df['body'].swifter.apply(get_10_bun_num)
    df['bun20_num'] = df['body'].swifter.apply(get_20_bun_num)
    df['bun30_num'] = df['body'].swifter.apply(get_30_bun_num)
    df['bun10_per'] = df['bun10_num'] / df['bun_num']
    df['bun20_per'] = df['bun20_num'] / df['bun_num']
    df['bun30_per'] = df['bun30_num'] / df['bun_num']

    df['body'] = df['body'].swifter.apply(normalize)  # rm rubi, line

    df['text_length'] = df['body'].str.len()
    df['comment_num'] = df['body'].swifter.apply(get_comment_num)
    df['time'] = df['body'].swifter.apply(lambda x: get_date(get_times(x)))
    df['times_flag'] = df['time'].swifter.apply(lambda x: 1 if x else 0)
    df['times_akutagawa'] = (df['time'] >= date(1910, 1, 1)) & (df['time'] <= date(1927, 7, 24))
    df['times_akutagawa'] = df['times_akutagawa'].swifter.apply(lambda x: 1 if x else 0)
    df['line_length_mean'] = df['body'].swifter.apply(lambda x: np.mean([len(l) for l in x.split('\n\r') if len(l) > 1]))
    df['line_length_max'] = df['body'].swifter.apply(lambda x: np.max([len(l) for l in x.split('\n\r') if len(l) > 1]))
    df['line_length_min'] = df['body'].swifter.apply(lambda x: np.min([len(l) for l in x.split('\n\r') if len(l) > 1]))
    df['line_length_std'] = df['body'].swifter.apply(lambda x: np.std([len(l) for l in x.split('\n\r') if len(l) > 1]))
    df['has_akutagawa'] = df['body'].swifter.apply(lambda x: 1 if '芥川龍之介' in x else 0)
    df['has_akutagawa_num'] = df['body'].swifter.apply(lambda x: x.count('芥川') if type(x) == str else 0)
    df['author_num'] = df['body'].swifter.apply(lambda x: sum([x.count(x) if type(x) == str else 0 for x in author_names]))
    df['akutagawa_works_num'] = df['body'].swifter.apply(lambda x: sum([x.count(x) if type(x) == str else 0 for x in akutagawa_works]))
    df['aozora_works_num'] = df['body'].swifter.apply(lambda x: sum([x.count(x) if type(x) == str else 0 for x in aozora_works]))
    df['hiragana_num'] = df['body'].swifter.apply(get_hiragana_num)
    df['kanji_num'] = df['body'].swifter.apply(get_kanji_num)
    df['katakana_num'] = df['body'].swifter.apply(get_katakana_num)
    df['tokusyu_num'] = df['body'].swifter.apply(get_tokusyu_num)
    df['kigou_num'] = df['body'].swifter.apply(get_kigou_num)
    df['end_num'] = df['body'].swifter.apply(get_end_num)
    df['alphabet_num'] = df['body'].swifter.apply(get_alpha_num)
    df['gimon_num'] = df['body'].swifter.apply(get_gimon_num)
    df['kantan_num'] = df['body'].swifter.apply(get_kantan_num)
    df['kakko_num'] = df['body'].swifter.apply(get_kakko_num)
    df['zenkakko_num'] = df['body'].swifter.apply(get_zenkakko_num)
    df['nobasi_num'] = df['body'].swifter.apply(get_nobasi_num)
    df['kaiwa_num'] = df['body'].swifter.apply(get_kaiwa_num)
    df['kaigyo_kaiwa_num'] = df['body'].swifter.apply(get_kaikaiwa_num)
    df['ten_num'] = df['body'].swifter.apply(get_ten_num)
    df['repeat_num'] = df['body'].swifter.apply(get_repeat_num)
    df['bar_num'] = df['body'].swifter.apply(get_bar_num)
    df['date_num'] = df['body'].swifter.apply(get_date_num)
    df['star_num'] = df['body'].swifter.apply(lambda x: x.count('★') if type(x) == str else 0)
    df['cross_num'] = df['body'].swifter.apply(lambda x: x.count('×') if type(x) == str else 0)
    df['square_num'] = df['body'].swifter.apply(lambda x: x.count('□') if type(x) == str else 0)
    df['circle_num'] = df['body'].swifter.apply(lambda x: x.count('○') if type(x) == str else 0)
    df['dot_num'] = df['body'].swifter.apply(lambda x: x.count('・') if type(x) == str else 0)
    df['kome_num'] = df['body'].swifter.apply(lambda x: x.count('※') if type(x) == str else 0)
    df['ast_num'] = df['body'].swifter.apply(lambda x: x.count('＊') if type(x) == str else 0)
    df['has_akutagawa_per'] = df['has_akutagawa_num'] / df['text_length']
    df['author_per'] = df['author_num'] / df['text_length']
    df['akutagawa_works_per'] = df['akutagawa_works_num'] / df['text_length']
    df['aozora_works_per'] = df['aozora_works_num'] / df['text_length']
    df['zenkuhaku_per'] = df['zenkuhaku_num'] / df['text_length']
    df['hiragana_per'] = df['hiragana_num'] / df['text_length']
    df['kanji_per'] = df['kanji_num'] / df['text_length']
    df['katakana_per'] = df['katakana_num'] / df['text_length']
    df['tokusyu_per'] = df['tokusyu_num'] / df['text_length']
    df['kigou_per'] = df['kigou_num'] / df['text_length']
    df['end_per'] = df['end_num'] / df['text_length']
    df['alphabet_per'] = df['alphabet_num'] / df['text_length']
    df['gimon_per'] = df['gimon_num'] / df['text_length']
    df['kantan_per'] = df['kantan_num'] / df['text_length']
    df['kakko_per'] = df['kakko_num'] / df['text_length']
    df['zenkakko_per'] = df['zenkakko_num'] / df['text_length']
    df['nobasi_per'] = df['nobasi_num'] / df['text_length']
    df['kaiwa_per'] = df['kaiwa_num'] / df['text_length']
    df['kaigyo_kaiwa_per'] = df['kaigyo_kaiwa_num'] / df['text_length']
    df['ten_per'] = df['ten_num'] / df['text_length']
    df['repeat_per'] = df['repeat_num'] / df['text_length']
    df['date_per'] = df['date_num'] / df['text_length']
    df['bar_per'] = df['bar_num'] / df['text_length']
    df['star_per'] = df['star_num'] / df['text_length']
    df['cross_per'] = df['cross_num'] / df['text_length']
    df['square_per'] = df['square_num'] / df['text_length']
    df['circle_per'] = df['circle_num'] / df['text_length']
    df['dot_per'] = df['dot_num'] / df['text_length']
    df['kome_per'] = df['kome_num'] / df['text_length']
    df['ast_per'] = df['ast_num'] / df['text_length']
    df['sp1_num'] = df['body'].swifter.apply(get_sp1_num)
    df['sp2_num'] = df['body'].swifter.apply(get_sp2_num)
    df['sp3_num'] = df['body'].swifter.apply(get_sp3_num)
    df['sp4_num'] = df['body'].swifter.apply(get_sp4_num)
    df['sp5_num'] = df['body'].swifter.apply(get_sp5_num)
    df['sp6_num'] = df['body'].swifter.apply(get_sp6_num)
    df['sp7_num'] = df['body'].swifter.apply(get_sp7_num)
    df['sp8_num'] = df['body'].swifter.apply(get_sp8_num)
    df['sp9_num'] = df['body'].swifter.apply(get_sp9_num)
    df['sp10_num'] = df['body'].swifter.apply(get_sp10_num)
    df['sp11_num'] = df['body'].swifter.apply(get_sp11_num)
    df['sp12_num'] = df['body'].swifter.apply(get_sp12_num)
    df['sp13_num'] = df['body'].swifter.apply(get_sp13_num)
    df['sp14_num'] = df['body'].swifter.apply(get_sp14_num)
    df['sp15_num'] = df['body'].swifter.apply(get_sp15_num)
    df['sp16_num'] = df['body'].swifter.apply(get_sp16_num)
    df['sp17_num'] = df['body'].swifter.apply(get_sp17_num)
    df['sp1_per'] = df['sp1_num'] / df['text_length']
    df['sp2_per'] = df['sp2_num'] / df['text_length']
    df['sp3_per'] = df['sp3_num'] / df['text_length']
    df['sp4_per'] = df['sp4_num'] / df['text_length']
    df['sp5_per'] = df['sp5_num'] / df['text_length']
    df['sp6_per'] = df['sp6_num'] / df['text_length']
    df['sp7_per'] = df['sp7_num'] / df['text_length']
    df['sp8_per'] = df['sp8_num'] / df['text_length']
    df['sp9_per'] = df['sp9_num'] / df['text_length']
    df['sp10_per'] = df['sp10_num'] / df['text_length']
    df['sp11_per'] = df['sp11_num'] / df['text_length']
    df['sp12_per'] = df['sp12_num'] / df['text_length']
    df['sp13_per'] = df['sp13_num'] / df['text_length']
    df['sp14_per'] = df['sp14_num'] / df['text_length']
    df['sp15_per'] = df['sp15_num'] / df['text_length']
    df['sp16_per'] = df['sp16_num'] / df['text_length']
    df['sp17_per'] = df['sp17_num'] / df['text_length']
    df['ku1_num'] = df['body'].swifter.apply(get_ku1_num)
    df['ku2_num'] = df['body'].swifter.apply(get_ku2_num)
    df['ku3_num'] = df['body'].swifter.apply(get_ku3_num)
    df['ku4_num'] = df['body'].swifter.apply(get_ku4_num)
    df['ku5_num'] = df['body'].swifter.apply(get_ku5_num)
    df['ku6_num'] = df['body'].swifter.apply(get_ku6_num)
    df['ku7_num'] = df['body'].swifter.apply(get_ku7_num)
    df['ku8_num'] = df['body'].swifter.apply(get_ku8_num)
    df['ku9_num'] = df['body'].swifter.apply(get_ku9_num)
    df['ku10_num'] = df['body'].swifter.apply(get_ku10_num)
    df['ku11_num'] = df['body'].swifter.apply(get_ku11_num)
    df['ku12_num'] = df['body'].swifter.apply(get_ku12_num)
    df['ku13_num'] = df['body'].swifter.apply(get_ku13_num)
    df['ku1_per'] = df['ku1_num'] / df['text_length']
    df['ku2_per'] = df['ku2_num'] / df['text_length']
    df['ku3_per'] = df['ku3_num'] / df['text_length']
    df['ku4_per'] = df['ku4_num'] / df['text_length']
    df['ku5_per'] = df['ku5_num'] / df['text_length']
    df['ku6_per'] = df['ku6_num'] / df['text_length']
    df['ku7_per'] = df['ku7_num'] / df['text_length']
    df['ku8_per'] = df['ku8_num'] / df['text_length']
    df['ku9_per'] = df['ku9_num'] / df['text_length']
    df['ku10_per'] = df['ku10_num'] / df['text_length']
    df['ku11_per'] = df['ku11_num'] / df['text_length']
    df['ku12_per'] = df['ku12_num'] / df['text_length']
    df['ku13_per'] = df['ku13_num'] / df['text_length']
    return df
