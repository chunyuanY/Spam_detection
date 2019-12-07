import re
import jieba
from datetime import datetime
import numpy as np
pattern = re.compile("[A-Za-z0-9]")
punct = re.compile("[\(\)\?~\|\!！\.。：、\:\^\+\-=？&\>\"「」》《￥$#@【】]")
space = re.compile("\s+")

def chartype(ch):
    if pattern.match(ch):
        return 'w'
    elif punct.match(ch):
        return 's'
    else:
        return 'cn'

def cut_char(text):
    word = ""
    sign = ""
    cn_text = ""
    newchar = ""
    char_list = []

    for ch in text:
        oldchar = newchar
        newchar = ch
        if pattern.match(newchar):
            word += ch
        elif punct.match(newchar):
            sign += ch
        else:
            cn_text += ch

        if chartype(oldchar)!=chartype(newchar):
            if chartype(oldchar) == 'w':
                char_list.append(word)
                word = ""
            elif chartype(oldchar) == 's':
                char_list.append(sign)
                sign = ""
            else:
                char_list.append(cn_text)
                cn_text = ""

    if word != "":
        char_list.append(word)
    if sign != "":
        char_list.append(sign)
    if cn_text != "":
        char_list.append(cn_text)

    content = " ".join(char_list)
    content = re.sub("\s+", " ", content)
    return content


def full2half(s):
    '''
    全角转半角
    测试用例："ｍｎ123abc好人"
    :param s:
    :return:
    '''
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)

def extract_emoji(content):
    pattern = re.compile("\|.{1,10}\|")
    return pattern.findall(content)

def extract_url(content):
    pattern = re.compile("http")
    return pattern.findall(content)

def extract_pronouns(content):
    pattern = re.compile("我|我們|你|你們|他|他們")
    return pattern.findall(content)


def extract_nouns(content):
    spam_pattern = re.compile("體驗|galaxy|note|nexus|sii|三星|功能|不過|遊戲|影片|大家|上市")
    spam_cnt = len(spam_pattern.findall(content))
    nonspam_pattern = re.compile("問題|解決|無法|開機|設定|怎麽|如何|正常|是否|小弟|大大|辦法|安裝|rom|聲音")
    nonspam_cnt = len(nonspam_pattern.findall(content))
    return spam_cnt, nonspam_cnt


def extract_word(text):
    word = ""
    char_list = []
    for ch in text:
        if pattern.match(ch):
            word += ch
        else:
            if not space.match(word) and word!="":
                char_list.append(word)
            if not space.match(ch) and ch!="":
                char_list.append(ch)
            word = ""
    if word != "":
        char_list.append(word)
    return char_list



def text_statistic_feature(content_list):
    features = []
    for content in content_list:
        length = len(content)

        n_all = len(content)
        n_words = len(list(jieba.cut(content)))
        n_lines = len(content.splitlines())
        n_hyperlinks = len(extract_url(content))
        n_emoticon = len(extract_emoji(content))

        if length != 0:
            p_digit = len(re.compile("[0-9]").findall(content)) / length
            p_english = len(re.compile("[a-zA-Z]").findall(content)) / length
            p_punct = len(re.compile("[,\./\<\>\?;'\\\\:\"\|\[\]\{\}\-=_\+`~，。、《》？；‘：“【】——！@#￥%……&*（）·!$^*\(\)]").findall(content)) / length
            p_special = len([c for c in content if not str(c).isalnum()]) / length
            p_wspace = len(re.compile("\s").findall(content)) / length
            p_immediacy = len(extract_pronouns(content)) / length
        else:
            p_digit, p_english, p_punct, p_special, p_wspace, p_immediacy, = 0,0,0,0,0,0

        features.append([n_all, n_words, n_lines, n_hyperlinks, n_emoticon, p_immediacy, p_digit, p_english, p_punct, p_special, p_wspace])
    return features


def time_feature(time_list):
    time_f = []
    for time in time_list:
        date = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
        hour = np.eye(24)[date.hour-1].tolist()
        weekday = np.eye(7)[date.weekday()].tolist()
        time_f.append(hour+weekday)
    return time_f
