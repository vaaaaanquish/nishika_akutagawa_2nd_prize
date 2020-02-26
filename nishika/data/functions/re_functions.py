import re
import regex
"""
//------------------
normalize関数群
//------------------
"""


def top_kuhaku(x):
    return 1 if '　' == x.replace('\r\n', '')[0] else 0


def normalize(text):
    text = re.split(r'底本：', text)[0]
    text = re.sub(r'\r\n', '', text)
    text = re.sub(r'《.*?》|［＃.*?］|｜', '', text)
    text = re.sub(r'［.*見出し］', '', text)
    return text.strip()


"""
//------------------
以下、正規表現系関数群
//------------------
"""

hp = regex.compile(r'\p{Hiragana}+')
kp = regex.compile(r'\p{Han}+')
ktp = regex.compile(r'\p{Katakana}+')
kansup = re.compile(r'[一二三四五六七八九十]+')
rp = re.compile(r'[ゐゑヱゝヽゞヾヴヹヷゐ゙ヸヰヺ]+')
kip = re.compile(r'[、]+')
endp = re.compile(r'[。]+')
ap = re.compile(r'[a-zA-Z\s]+')
kap = re.compile(r'[!！]+')
gip = re.compile(r'[?？]+')
senp = re.compile(r'[―]+')
tenp = re.compile(r'[…]+')
kaip = re.compile(r'[。！？―…]」')
kaiwap = re.compile(r'」\r\n「')
kurip = re.compile(r'[／″＼]+')
barp = re.compile(r'[|]+')
kuhakup = re.compile(r'[\u3000\s　]+')
headp = re.compile(r'\r\n\r\n[^\r\n\u3000\s　]+\r\n\r\n')
datep = re.compile(r'[一二三四五六七八九十]+月[一二三四五六七八九十]+日')
bunp = re.compile(r'\r\n[^\r\n]+\r\n')
suip = re.compile(r'第.水準')
tenkaip = re.compile(r'、\r\n')
kukaip = re.compile(r'\r\n　')
kurikp = re.compile(r'[ゝヽゞヾ]+')

ku1 = re.compile(r'西洋')
ku2 = re.compile(r'ロシア')
ku3 = re.compile(r'日本')
ku4 = re.compile(r'フランス')
ku5 = re.compile(r'ドイツ')
ku6 = re.compile(r'イタリア')
ku7 = re.compile(r'イギリス')
ku8 = re.compile(r'英吉利')
ku9 = re.compile(r'英国')
ku10 = re.compile(r'独逸')
ku11 = re.compile(r'法蘭西')
ku12 = re.compile(r'露西亜')
ku13 = re.compile(r'伊太利')

sp1 = re.compile(r'僕')
sp2 = re.compile(r'私')
sp3 = re.compile(r'彼')
sp4 = re.compile(r'彼女')
sp5 = re.compile(r'自分')
sp6 = re.compile(r'君')
sp7 = re.compile(r'娘')
sp8 = re.compile(r'母')
sp9 = re.compile(r'父')
sp10 = re.compile(r'俺.')
sp11 = re.compile(r'先生')
sp12 = re.compile(r'親分')
sp13 = re.compile(r'子供')
sp14 = re.compile(r'子')
sp15 = re.compile(r'主人')
sp16 = re.compile(r'氏')
sp17 = re.compile(r'婦人')


def get_comment_num(x):
    regex = re.compile("「.*?」", flags=re.DOTALL)
    matchArray = regex.findall(x)
    return len(matchArray)


def get_rubi_num(x):
    regex = re.compile("《.*?》", flags=re.DOTALL)
    matchArray = regex.findall(x)
    return len(matchArray)


def get_kakko_num(x):
    regex = re.compile("(.*?)", flags=re.DOTALL)
    matchArray = regex.findall(x)
    return len(matchArray)


def get_zenkakko_num(x):
    regex = re.compile("（.*?）", flags=re.DOTALL)
    matchArray = regex.findall(x)
    return len(matchArray)


def get_line_num(x):
    regex = re.compile("\r\n", flags=re.DOTALL)
    matchArray = regex.findall(x)
    return len(matchArray)


def get_lines_num(x):
    regex = re.compile("\r\n\r\n", flags=re.DOTALL)
    matchArray = regex.findall(x)
    return len(matchArray)


def get_10_bun_num(x):
    c = 0
    for w in bunp.findall(x):
        w = re.sub(r'《.*?》|［＃.*?］|｜', '', w).strip()
        if len(w) > 2 and len(w) < 10:
            c += 1
    return c


def get_20_bun_num(x):
    c = 0
    for w in bunp.findall(x):
        w = re.sub(r'《.*?》|［＃.*?］|｜', '', w).strip()
        if len(w) > 10 and len(w) < 30:
            c += 1
    return c


def get_30_bun_num(x):
    c = 0
    for w in bunp.findall(x):
        w = re.sub(r'《.*?》|［＃.*?］|｜', '', w).strip()
        if len(w) > 30:
            c += 1
    return c


def get_suijun_num(x):
    return len(suip.findall(x))


def get_tenkai_num(x):
    return len(tenkaip.findall(x))


def get_kukai_num(x):
    return len(kukaip.findall(x))


def get_kurik_num(x):
    return len(kurikp.findall(x))


def get_bun_num(x):
    return len(bunp.findall(x))


def get_date_num(x):
    return len(datep.findall(x))


def get_kuhaku_num(x):
    return len(kuhakup.findall(x))


def get_bar_num(x):
    return len(barp.findall(x))


def get_kansu_num(x):
    return len(kansup.findall(x))


def get_repeat_num(x):
    return len(kurip.findall(x))


def get_ten_num(x):
    return len(tenp.findall(x))


def get_kaiwa_num(x):
    return len(kaip.findall(x))


def get_kaikaiwa_num(x):
    return len(kaiwap.findall(x))


def get_nobasi_num(x):
    return len(senp.findall(x))


def get_kantan_num(x):
    return len(kap.findall(x))


def get_gimon_num(x):
    return len(gip.findall(x))


def get_hiragana_num(x):
    return len(hp.findall(x))


def get_kanji_num(x):
    return len(kp.findall(x))


def get_katakana_num(x):
    return len(ktp.findall(x))


def get_tokusyu_num(x):
    return len(rp.findall(x))


def get_kigou_num(x):
    return len(kip.findall(x))


def get_end_num(x):
    return len(endp.findall(x))


def get_alpha_num(x):
    return len(ap.findall(x))


def get_zenkaku_kuhaku_num(x):
    return x.count('　') if type(x) == str else 0


def get_ku1_num(x):
    return len(ku1.findall(x))


def get_ku2_num(x):
    return len(ku2.findall(x))


def get_ku3_num(x):
    return len(ku3.findall(x))


def get_ku4_num(x):
    return len(ku4.findall(x))


def get_ku5_num(x):
    return len(ku5.findall(x))


def get_ku6_num(x):
    return len(ku6.findall(x))


def get_ku7_num(x):
    return len(ku7.findall(x))


def get_ku8_num(x):
    return len(ku8.findall(x))


def get_ku9_num(x):
    return len(ku9.findall(x))


def get_ku10_num(x):
    return len(ku10.findall(x))


def get_ku11_num(x):
    return len(ku11.findall(x))


def get_ku12_num(x):
    return len(ku12.findall(x))


def get_ku13_num(x):
    return len(ku13.findall(x))


def get_sp1_num(x):
    return len(sp1.findall(x))


def get_sp2_num(x):
    return len(sp2.findall(x))


def get_sp3_num(x):
    return len(sp3.findall(x))


def get_sp4_num(x):
    return len(sp4.findall(x))


def get_sp5_num(x):
    return len(sp5.findall(x))


def get_sp6_num(x):
    return len(sp6.findall(x))


def get_sp7_num(x):
    return len(sp7.findall(x))


def get_sp8_num(x):
    return len(sp8.findall(x))


def get_sp9_num(x):
    return len(sp9.findall(x))


def get_sp10_num(x):
    return len(sp10.findall(x))


def get_sp11_num(x):
    return len(sp11.findall(x))


def get_sp12_num(x):
    return len(sp12.findall(x))


def get_sp13_num(x):
    return len(sp13.findall(x))


def get_sp14_num(x):
    return len(sp14.findall(x))


def get_sp15_num(x):
    return len(sp15.findall(x))


def get_sp16_num(x):
    return len(sp16.findall(x))


def get_sp17_num(x):
    return len(sp17.findall(x))


def get_heading_num(x):
    return len([w for w in headp.findall(x) if re.sub(r'《.*?》|［＃.*?］|｜', '', w).strip()])
