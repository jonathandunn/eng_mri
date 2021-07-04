# eng_mri

This package identifies code switching between English (ENG) and te reo Māori (MRI). For each word, a value toward 1 indicates increased probability that a particular item is from MRI.


# Usage

Import and initialize the package:

	from eng_mri import LID
	LID = LID()
	
Given a string, get the probability that each word is in MRI (and the probability for the whole string):

	words, overall = LID.predict(line)
	
# Example Output

[('more', 0.123798005), ('jobs', 0.046649054), ('in', 0.016111122), ('northland', 0.093827195), ('he', 0.7417166), ('mahi', 0.9320169), ('ano', 0.7113533), ('kia', 0.93088806), ('ora', 0.80610853), ('whanau', 0.9679866)]

[('with', 0.01944267), ('thanks', 0.011764448), ('to', 0.2852968), ('reo', 0.9274818), ('irirangi', 0.9879665), ('o', 0.77238846), ('tamakimakaurau', 0.99814606), ('ko', 0.95399827), ('waatea', 0.8636462), ('te', 0.88383573), ('ingoa', 0.8135497), ('good', 0.12488466), ('news', 0.01428419), ('for', 0.23447321), ('taitokerau', 0.99377984), ('tena', 0.9337327), ('koutou', 0.9862619), ('rau', 0.92139596), ('rangatira', 0.9941089), ('ma', 0.72965074)]

	
# Installation

	pip install git+https://github.com/jonathandunn/eng_mri.git
