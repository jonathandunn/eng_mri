import os
import re
import codecs
import cleantext
import operator
from joblib import dump, load
from pathlib import Path
import tensorflow as tf
import numpy as np
import cytoolz as ct
from sklearn.feature_extraction.text import CountVectorizer

#----------------------------------------------------
class Tokenizer(object):
	
	def __init__(self, sample_size = 20):
		self.sample_size = sample_size

	##-------
	## This function controls string cleaning
	##-------
	def clean_line(self, line):
			
		line = cleantext.clean(line,
				fix_unicode = True,
				to_ascii = False,
				lower = True,
				no_line_breaks = True,
				no_urls = True,
				no_emails = True,
				no_phone_numbers = True,
				no_numbers = True,
				no_digits = True,
				no_currency_symbols = True,
				no_punct = True,
				replace_with_url = "",
				replace_with_email = "",
				replace_with_phone_number = "",
				replace_with_number = "",
				replace_with_digit = "",
				replace_with_currency_symbol = ""
				)
				
		cleanr1 = re.compile('<.*?>')
		cleanr2 = re.compile('<.*?\s')
		
		line = re.sub(cleanr1, '', line)
		line = line.replace("\\", "<")
		line = re.sub(cleanr2, '', line)

		return line.lstrip().rstrip()
#----------------------------------------------------

class LID(object):

	#---------
	def __init__(self, model_name = "v20.10.NZ.Trigrams_v2.25k.50x2_layers", sample_size = 20):

		self.langs_total = ["eng", "mri"]
		
		#Create dictionary for labels
		self.label_to_lang = {}
		for i in range(len(self.langs_total)):
			self.label_to_lang[i] = self.langs_total[i]
		self.lang_to_label = {v: k for k, v in self.label_to_lang.items()}
		self.n_labels = len(self.lang_to_label.keys())

		#Set class variables
		self.sample_size = sample_size
		self.model_name = model_name
		self.model_file = Path(__file__).parent / os.path.join(".", self.model_name)
		
		#Set feature name
		if "25k" in model_name:
			self.feature_name = "trigrams.NZ.25k.v2.joblib"
		
		#Initialize encoder
		from .eng_mri import Tokenizer
		self.Tokenizer = Tokenizer(sample_size = self.sample_size)
		
		#Load the pre-computed vocabulary
		self.vocabulary_set = load(Path(__file__).parent / os.path.join(".", self.feature_name))
		self.vocab_size = len(self.vocabulary_set)

		with open(os.path.join(".","pacificLID","pacificLID", self.feature_name), "rb") as fo:
			features = load(fo)
			self.encoder = CountVectorizer(
					input = "content", 
					encoding = "utf-8", 
					decode_error = "ignore", 
					strip_accents = None, 
					lowercase = True, 
					tokenizer = None,
					ngram_range = (3, 3), 
					analyzer = "char_wb", 
					vocabulary = features, 
					binary = False
					)
			
			self.encoder.vocab_size = len(features)
			self.encoder.encode = self.encoder.transform

		#Load pre-trained model
		self.model = tf.keras.models.load_model(Path(__file__).parent / os.path.join(".", self.model_file))

		return

	#----------
	def predict(self, line):
	
		#Clean it
		line = self.Tokenizer.clean_line(line)
		
		#Get all sub-sequences
		window_pre = [line[:i] for i in range(1,10)]
		window = ["".join(x) for x in ct.sliding_window(20, line)]
		window_post = [line[-i:] for i in range(1,11)]
			
		#Predict over each sub-sequence
		probabilities = self.model.predict(self.encoder.encode(window_pre+window+window_post))
			
		#Get word indexes
		words = [m.start() for m in re.finditer("\s", line)]
		words = list(ct.sliding_window(2, words)) + [(words[-1],len(line))]
			
		#Get word and summed probability
		text = [line[word[0]:word[1]] for word in words]
		label = ["mri" if probabilities[word[0]:word[1]].mean(axis = 0)[1] > 0.5 else "eng" for word in words]
			
		return text, label
	
	#----------