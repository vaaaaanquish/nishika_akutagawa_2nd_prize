from kanjize import kanji2int
from datetime import datetime, date
from jeraconv import jeraconv
import re
"""
文末の時間情報を取得、パースするやつ
"""

tt_ksuji = str.maketrans('一二三四五六七八九〇壱弐参', '1234567890123')

re_suji = re.compile(r'[十拾百千万億兆\d]+')
re_kunit = re.compile(r'[十拾百千]|\d+')
re_manshin = re.compile(r'[万億兆]|[^万億兆]+')

TRANSUNIT = {'十': 10, '拾': 10, '百': 100, '千': 1000}
TRANSMANS = {'万': 10000, '億': 100000000, '兆': 1000000000000}


def kansuji2arabic(kstring: str, sep=False):
    def _transvalue(sj: str, re_obj=re_kunit, transdic=TRANSUNIT):
        unit = 1
        result = 0
        for piece in reversed(re_obj.findall(sj)):
            if piece in transdic:
                if unit > 1:
                    result += unit
                unit = transdic[piece]
            else:
                val = int(piece) if piece.isdecimal() else _transvalue(piece)
                result += val * unit
                unit = 1

        if unit > 1:
            result += unit

        return result

    transuji = kstring.translate(tt_ksuji)
    for suji in sorted(set(re_suji.findall(transuji)), key=lambda s: len(s), reverse=True):
        if not suji.isdecimal():
            arabic = _transvalue(suji, re_manshin, TRANSMANS)
            arabic = '{:,}'.format(arabic) if sep else str(arabic)
            transuji = transuji.replace(suji, arabic)
        elif sep and len(suji) > 3:
            transuji = transuji.replace(suji, '{:,}'.format(int(suji)))

    return transuji


def get_times(x):
    x = x.strip().split('\n')[-1]
    regex = re.compile("（.*?）", flags=re.DOTALL)
    matchArray = regex.findall(x)
    if matchArray:
        return kansuji2arabic(matchArray[0])
    return ''


def get_date(x):
    x = x.replace('（', '').replace('）', '')
    if '年' in x:
        ys = x.split('年')[0]
        try:
            j2w = jeraconv.J2W()
            y = j2w.convert(ys + '年')
        except:
            try:
                y = int(ys)
            except:
                return None
        md = x.split('年')[1]
        m = 1
        d = 1
        if '月' in md and '日' in md:
            date_pattern = re.compile('(\d{1,2})月(\d{1,2})日')
            result = date_pattern.search(md)
            if result:
                m, d = result.groups()
        elif '月' in md:
            date_pattern = re.compile('(\d{1,2})月')
            result = date_pattern.search(md)
            if result:
                m = result.groups()[0]
        return date(int(y), int(m), int(d))

    elif '・' in x:
        ys = x.split('・')[0]
        try:
            j2w = jeraconv.J2W()
            y = j2w.convert(ys + '年')
        except:
            try:
                y = int(ys)
                if y < 100:  #(15,1,1)
                    return None
            except:
                return None
        md = '・'.join(x.split('・')[1:])
        m = 1
        d = 1
        if '・' in md:
            date_pattern = re.compile('(\d{1,2})・(\d{1,2})')
            result = date_pattern.search(md)
            if result:
                m, d = result.groups()
        else:
            date_pattern = re.compile('(\d{1,2})')
            result = date_pattern.search(md)
            if result:
                m = result.groups()[0]
        return date(int(y), int(m), int(d))

    elif '、' in x:
        ys = x.split('、')[0]
        try:
            j2w = jeraconv.J2W()
            y = j2w.convert(ys + '年')
        except:
            try:
                y = int(ys)
                if y < 100:  #(15,1,1)
                    return None
            except:
                return None
        md = '、'.join(x.split('、')[1:])
        m = 1
        d = 1
        if '、' in md:
            date_pattern = re.compile('(\d{1,2})、(\d{1,2})')
            result = date_pattern.search(md)
            if result:
                m, d = result.groups()
        else:
            date_pattern = re.compile('(\d{1,2})')
            result = date_pattern.search(md)
            if result:
                m = result.groups()[0]
        return date(int(y), int(m), int(d))

    return None
