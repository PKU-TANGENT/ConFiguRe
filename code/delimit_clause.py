import re
from functools import reduce
all_clause_patterns = (
    '([！!。?？])([^”’])',# 单字符断句符 single letter termination punctuation
    '(\.{6})([^”’])',# 英文省略号 ellipses
    '(\…{2})([^”’])',# 中文省略号 Chinese ellipses
    '(\—{2})([^”’])',# 中文破折号 Chinese dash
    '([!?！？。—…][”’])([^”’])',# 如果后面有后引号则以后引号断开 incorporate the latter quotation mark of a pair, should it follow a termination punctuation
    '([,，；：;:])([^“‘])',# 单字符断句符 in-sentence punctuation
    '([,，；：;:][“‘])([^”’])'# 如果句中分割后有前引号则以前引号后分割。 incorporate the former quotation mark of a pair, should it follow an in-sentence punctuation
)
delimit_clause_helper_func = lambda para, pattern: re.sub(pattern, r"\1\n\2", para)
def delimit_clause(para):
    para = reduce(delimit_clause_helper_func, all_clause_patterns, para)
    para = para.rstrip() # 段尾如果有多余的\n就去掉它 remove redundant \n
    return para.split("\n")
# Note: "delimit_clause" condensed several rules, and is now rewritten in `reduce` style.
# Rules are generally concurred. One reference would be https://blog.csdn.net/blmoistawinde/article/details/82379256

"""
第一种分割，将单字符短句符分割开，包括。？?！!省略号......  破折号————
如果其后有后引号则以后引号为根据断开
第二种分割，句子内部分割 包括 逗号， ,分号；; 冒号：:
如果其后有前引号，则在前引号后断开（因为标注时会在前引号后面作为）
"""

"""
for testing purposes
data=["你说：“如果不是你,我会在哪里——”","“你看！”她说,“这才是我！”","今天星期一,不用去上学。","只是淡淡的说的：“劝君更尽一杯酒，西出阳关无故人。”"]
try:
    assert list(map(delimit_clause, data)) == list(map(old_delimit_clause, data))
except:
    print(list(map(delimit_clause, data)))
    print(list(map(old_delimit_clause, data)))
"""
