# -*-coding:utf-8 -*-


def bilingual_evaluation_understudy_score(pred, target):
    """
    bilingual evaluation understudy (BLEU) score

    성과지표로 데이터의 X가 순서정보를 가진 단어들(문장)로 이루어져 있고,
    y또한 단어들의 시리즈(문장) 로 이루어진 경우에 사용되며, 변역을 하는 모델의 주로 사용이 된다.

    - n-gram 읕 통한 순써상들이 얼마나 겹치는지 측정(precision)
    - 문장길이에 대한 과적합 보정(Brevity Penalty)
    - 같은 단어가 연속적으로 나올떄 과적합 되는 것을 보정(Clipping)

    """
    pass
