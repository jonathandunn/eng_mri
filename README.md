# eng_mri

This package identifies code switching between English (ENG) and te reo Māori (MRI). For each word, a value toward 1 indicate increased probability that a particular item is from MRI.


# Usage

Import and initialize the package:

	from eng_mri import LID
	LID = LID()
	
Given a string, get the probability that each word is in MRI (and the probability for the whole string):

	words, overall = LID.predict(line)
	
# Installation

	pip install git+https://github.com/jonathandunn/eng_mri.git