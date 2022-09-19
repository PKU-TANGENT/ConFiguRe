import re
from functools import reduce
all_cut_patterns = ('([！!。?？])([^”’])','(\.{6})([^”’])','(\…{2})([^”’])','(\—{2})([^”’])','([!?！？。—…][”’])([^”’])','([,，；：;:])([^“‘])','([,，；：;:][“‘])([^”’])')
cut_sent_helper_func = lambda para, pattern: re.sub(pattern, r"\1\n\2", para)
def cut_sent(para):
    para = reduce(cut_sent_helper_func, all_cut_patterns, para)
    para = para.rstrip()
    return para.split("\n")
# Note: func `old_cut_sent` condensed several rules, and is now rewritten in `reduce` style.
# Rules are generally concurred. One reference would be https://blog.csdn.net/blmoistawinde/article/details/82379256
# def old_cut_sent(para):
#     para = re.sub('([！!。?？])([^”’])', r"\1\n\2", para)  # 单字符断句符 single letter termination punctuation
#     para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号 ellipses
#     para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号 Chinese ellipses
#     para = re.sub('(\—{2})([^”’])', r"\1\n\2", para)  # 中文破折号 Chinese dash
#     para = re.sub('([!?！？!。(——)(……)?][”’])([^”’])',
#                   r"\1\n\2", para)  # 如果后面有后引号则以后引号断开 incorporate the latter quotation mark of a pair, should it follow a termination punctuation
#     para = re.sub('([,，；：;:])([^“‘])', r"\1\n\2", para)  # 单字符断句符 in-sentence punctuation
#     para = re.sub('([,，；：;:][“‘])([^”’])', r"\1\n\2", para)
#     # 如果句中分割后有前引号则以前引号后分割。 incorporate the former quotation mark of a pair, should it follow an in-sentence punctuation
#     para = para.rstrip()  # 段尾如果有多余的\n就去掉它 remove redundant \n
#     return para.split("\n")


"""
第一种分割，将单字符短句符分割开，包括。？?！!省略号......  破折号————
如果其后有后引号则以后引号为根据断开
第二种分割，句子内部分割 包括 逗号， ,分号；; 冒号：:
如果其后有前引号，则在前引号后断开（因为标注时会在前引号后面作为）
"""


def cut_len(para, begin):
    para = para[0:begin]
    para = re.sub('([！!。?？])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('(\—{2})([^”’])', r"\1\n\2", para)  # 中文破折号
    para = re.sub('([!?！？!。(——)(……)?][”’])([^”’])',
                  r'\1\n\2', para)  # 如果后面有后引号则以后引号断开
    para = re.sub('([,，；：;:])([^“‘])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('([,，；：;:][“‘])([^”’])', r"\1\n\2", para)
    # 如果句中分割后有前引号则以前引号后分割。
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return len(para.split("\n"))


# data=["你说：“如果不是你,我会在哪里——”","“你看！”她说,“这才是我！”","今天星期一,不用去上学。","只是淡淡的说的：“劝君更尽一杯酒，西出阳关无故人。”"]
# try:
#     assert list(map(cut_sent, data)) == list(map(old_cut_sent, data))
# except:
#     print(list(map(cut_sent, data)))
#     print(list(map(old_cut_sent, data)))
