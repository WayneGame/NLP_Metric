#from transformers import GPT2LMHeadModel, GPT2Tokenizer
#from transformers import BertModel, BertTokenizer, BertConfig
#from transformers import DebertaTokenizer, DebertaModel, DebertaConfig
from transformers import *
import torch
from score_utils_2 import word_mover_score, lm_perplexity
from sentence_transformers import SentenceTransformer


class XMOVERScorer:

    def __init__(
        self,
        model_name=None,
        lm_name=None,
        do_lower_case=False,      
        device='cuda:0'
    ):

        if model_name == 'bert-base-multilingual-cased' :
            config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            self.model = BertModel.from_pretrained(model_name, config=config)
            self.model.to(device)
        elif model_name == 'distilbert-base-multilingual-cased':
            config = DistilBertConfig.from_pretrained(model_name,output_hidden_states=True, output_attentions=True)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            self.model = DistilBertModel.from_pretrained(model_name, config=config)
            self.model.to(device)
        elif model_name == 'sentence-transformers/paraphrase-xlm-r-multilingual-v1':
            config = XLMRobertaConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            self.model = XLMRobertaModel.from_pretrained(model_name, config=config)
            self.model.to(device)

        elif model_name == 'sentence-transformers/paraphrase-TinyBERT-L6-v2':
            config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            self.model = AutoModel.from_pretrained(model_name, config=config)
            self.model.to(device)
        else :
            if model_name== 'Tiny1':
                mod = 'E:\\Uni\\Aufnahmen Vorlesung\\SoSe21\Meta\\Gruppen_Repo\\NLP_Metric\\XMoverScore\\models\\TinyBERT-Repo\pre_trained\\6L_768D_FinalModel\\CoLA'

            config = BertConfig.from_pretrained(mod, output_hidden_states=True, output_attentions=True)
            self.tokenizer = BertTokenizer.from_pretrained(mod, do_lower_case=do_lower_case)
            self.model = BertModel.from_pretrained(mod, config=config)
            self.model.to(device)
        '''
        else:
            config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
            self.model = AutoModel.from_pretrained(model_name, config=config)
            self.model.to(device)
        '''



        if lm_name == "gpt2":
            self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
            self.lm.to(device)




    def compute_xmoverscore(self, mapping, projection, bias, source, translations, ngram=2, bs=32, layer=8, dropout_rate=0.3):
        return word_mover_score(mapping, projection, bias, self.model, self.tokenizer, source, translations, \
                                n_gram=ngram, layer=layer, dropout_rate=dropout_rate, batch_size=bs)
                     
    def compute_perplexity(self, translations, bs):        
        return lm_perplexity(self.lm, translations, self.lm_tokenizer, batch_size=bs)            
