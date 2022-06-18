from transformers import MarianMTModel, MarianTokenizer
import torch
import googletrans
import parlai.utils.logging as logging
from googletrans import Translator
from langdetect import detect, DetectorFactory


""" Language detection """
def detect_language(input_sent):
    """ 
    Given a sentence, return the language id of the detected langaug 
    """
    DetectorFactory.seed = 0 # for keeping the result consistent
    detected_language = detect(input_sent) 
    print(f"Detected {detected_language}!")
    return detected_language 


class MarianTranslator(object):
    def __init__(self, input_sentence):
        self.input_sentence = input_sentence
        self.model_name = None
       # if self.input_sentence is not None:
        self.src_lang = detect_language(self.input_sentence) 
        self.output_sentence = None
        
    
class MarianTranslatorFromTargetToEN(MarianTranslator):
    """
    Class to perform machine translation 
    from target language to English on a given sentence.
    """
    def __init__(self, MarianTranslator):
        ### Input params ###
        self.input_sentence = MarianTranslator.input_sentence
        self.model_name =  MarianTranslator.model_name
        self.src_lang = MarianTranslator.src_lang

        # for translating the input from the given language to English
        self.src_to_en_dict = {
            'ar': "Helsinki-NLP/opus-mt-ar-en",
            'bn': "Helsinki-NLP/opus-mt-bn-en",
            'ja': "Helsinki-NLP/opus-mt-ja-en",
            'ko': "Helsinki-NLP/opus-mt-ko-en",
            'fi': "Helsinki-NLP/opus-mt-fi-en",
        }
    
    def get_model_name(self):
        try:
            if self.src_lang in self.src_to_en_dict:
                self.model_name = self.src_to_en_dict[self.src_lang]
                print(f"Got {self.src_to_en_dict[self.src_lang]} as the Marian NMT model!")
                return self.model_name
        except:
            AttributeError("Cannot find a matching language in the Source to English Language Dictionary.")  
               
    def translate_question(self, input_sentence):
        self.input_sentence = input_sentence
        model = MarianMTModel.from_pretrained(self.get_model_name())
        tokenizer = MarianTokenizer.from_pretrained(self.get_model_name())
        print("Successfully initalize the model and tokenizer!")
        print("Now translating the sentence!")
        encoded_txt = tokenizer(self.input_sentence, return_tensors="pt",
                                padding=True)
        tgt_text = model.generate(**encoded_txt)
        tgt_text = tokenizer.batch_decode(tgt_text, skip_special_tokens=True)
        print(f"Translation from {self.src_lang} to en complete!")
        return tgt_text
   

class MarianTranslatorFromENToTarget(MarianTranslatorFromTargetToEN):
    """
    Class to perform machine translation 
    from target language to English on a given sentence.
    """
    
    def __init__(self, MarianTranslatorFromTargetToEN):
        ### Input params ###
        self.model_name =  None
        self.src_lang = MarianTranslatorFromTargetToEN.src_lang
        self.output_sentence = None

        # for translating the generated output from English to the target language
        self.en_to_tgt_dict = {
            'ar': "Helsinki-NLP/opus-mt-en-ar",
            'bn': None, # not available
            'ja': "Helsinki-NLP/opus-mt-en-jap",
            'ko': None, # not available
            'fi': "Helsinki-NLP/opus-mt-en-fi",
        }

    def get_model_name_for_answers(self):
        try:
            print(f"this is self.src_lang: {self.src_lang}")
            if self.src_lang in self.en_to_tgt_dict:
                if self.en_to_tgt_dict[self.src_lang] is not None:
                    self.model_name = self.en_to_tgt_dict[self.src_lang]
                    print(f"Got {self.en_to_tgt_dict[self.src_lang]} as the Marian NMT model!")
                    return self.model_name 
                else:
                    print(f"There is no available MarianNMT model for {self.src_lang}.")
                    print("You should swtich to Google Translator")
                    return None
        except:
            AttributeError("Cannot find a matching language in the English to Source Language Dictionary.")
        
   
    def translate_answer(self, output_sentence): #, model, tokenizer,
        self.output_sentence = output_sentence
        
        if self.get_model_name_for_answers() is not None: 
            model = MarianMTModel.from_pretrained(self.get_model_name_for_answers())
            tokenizer = MarianTokenizer.from_pretrained(self.get_model_name_for_answers())
            print("Successfully initalize the model and tokenizer!")
            print("Now translating the sentence!")
            encoded_txt = tokenizer(self.output_sentence, return_tensors="pt",
                                    padding=True)
            tgt_text = model.generate(**encoded_txt)
            tgt_text = tokenizer.batch_decode(tgt_text, skip_special_tokens=True)
            print(f"Translation from en to {self.src_lang} complete!")
            return tgt_text
        else:
            """ Google translator for bn-en and ko-en """
            print(f"Using Google Translator for en-{self.src_lang} translation instead")
            translator = Translator()
            translated = translator.translate(self.output_sentence, dest=self.src_lang)
            print(f"Translation from en to {self.src_lang} complete!")
            return translated.text