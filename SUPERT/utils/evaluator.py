from rouge.rouge import Rouge
from resources import *
from collections import OrderedDict
import os

from pythonrouge.pythonrouge import Pythonrouge

BASE_DIR =  os.getcwd() + "/SUPERT/" #os.path.dirname(os.path.abspath(__file__)) + "/../"
ROUGE_DIR = os.path.join(BASE_DIR,'rouge','ROUGE-RELEASE-1.5.5/') #do not delete the '/' in the end

def add_result(all_dic,result):
    for metric in result:
        if metric in all_dic:
            all_dic[metric].append(result[metric])
        else:
            all_dic[metric] = [result[metric]]


def evaluate_summary_rouge(cand,model,max_sum_len=100):
    rouge_scorer = Rouge(ROUGE_DIR,BASE_DIR,True)
    r1, r2, rl, rsu4 = rouge_scorer(cand,[model],max_sum_len)
    rouge_scorer.clean()
    dic = OrderedDict()
    dic['ROUGE-1'] = r1
    dic['ROUGE-2'] = r2
    dic['ROUGE-L'] = rl
    dic['ROUGE-SU4'] = rsu4
    return dic

    '''
    rouge_scorer = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=reference,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=max_sum_len,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge_scorer.calc_score()
    print(score)
    return score
    '''

