# -*- coding: utf-8 -*-
"""
Created on Jun 26, 2024

@author: Lian Jian Xiang

From Machine Learning for Materials (ML4M) Bootcamp -- July 18-20 2023
with Dr. Ben Afflerbach and Dr. Maciej Polak, Computational Materials Group,
University of Wisconsin–Madison.

Check:
"Flexible, Model-Agnostic Method for Materials Data Extraction from Text
 Using General Purpose Language Models"
Maciej P. Polak et al., Digital Discovery 2024, 3, 1221-1235
https://doi.org/10.1039/D4DD00016A

"""

# Prepare the notebook to look nicer
# from IPython.display import HTML, display

# def set_css():
#   display(HTML('''
#   <style>
#     pre {
#         white-space: pre-wrap;
#     }
#   </style>
#   '''))
# get_ipython().events.register('pre_run_cell', set_css)

# !pip install -q transformers sentencepiece

import transformers
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Change the tokenizer if needed
# from transformers import DebertaTokenizer
# tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

text = 'Welcome to ML Bootcamp!'
tokens = tokenizer.tokenize(text)
print(tokens)
token_ids = tokenizer.encode(text)
print(token_ids)

for t in token_ids:
    print(f"Token: {t} \t Decoded: {tokenizer.decode(t).replace(' ','')}")


###############################################################################

"""
 1. Classifying text to extract data

Working with traditional Language Models (LM)

Huggingface provides simple interfaces for LMs and Large LMs (LLMs)
https://huggingface.co/models

"""

from transformers import pipeline

# Function to test a text against a label/class - classification
def classify(text, label):
    classifier = pipeline("zero-shot-classification",
                          model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    sequence_to_classify = text
    candidate_labels = label
    output = classifier(sequence_to_classify,
                        candidate_labels,
                        multi_label=False)
    return(output["scores"], output["labels"])

# Test the function on simple text examples:
texts = [
    "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
    "Shakespeare wrote Romeo and Juliet.",
    "The American Civil War ended in 1865."]

print(classify(texts[0], ['science', 'literature', 'history']))
print(classify(texts[1], ['science', 'literature', 'history']))
print(classify(texts[2], ['science', 'literature', 'history']))

print(classify(texts[0], ['science']))
print(classify(texts[1], ['science']))
print(classify(texts[2], ['science']))


def single_classifier(text, label):
    if classify(text, [label])[0][0] > 0.5:
        return(True)
    else:
        return(False)

is_science = []
for t in texts:
    is_science.append(single_classifier(t, 'science'))
print(is_science)


# !pip install -q feedparser # lets us get papers from arXiv
import nltk
import feedparser # to access arXiv data
nltk.download('punkt') # needed to split text into sentences

# This function will search arXiv and return abstracts of research papers
# and it will use NLTK to split these texts into separate sentences
def search_arxiv(search_query, start=0, max_results=100):
    base_url = "http://export.arxiv.org/api/query?"
    query = "search_query=all:" + '+AND+all:'.join(f'"{s}"'.replace(' ', '+')
                                                   for s in search_query)
    print(query)
    papers = []

    # Get the total results for the search query
    feed = feedparser.parse(base_url + query)
    total_results = int(feed.feed.opensearch_totalresults)
    print(f"Total number of papers: {total_results}", end="\n\n")
    while start < total_results:
        page_query = query + f"&start={start}&max_results={max_results}"
        feed = feedparser.parse(base_url + page_query)
        for entry in feed.entries:
            paper = {
                "title": ' '.join(entry.title.split()),
                "doi": entry.id,
                "abstract": nltk.tokenize.sent_tokenize(' '.join(entry.summary.split())),
            }
            papers.append(paper)
        start += max_results
    return papers


# We'll search for the property of bulk modulus, it's a property of a material
# that tells us about how easily the material is compressed.
# Its units are units of pressure, i.e, GPa.
# For example aluminum has B=70 GPa. Diamond has B=500GPa

papers = search_arxiv(['bulk modulus', 'crystalline'])

# Printing the titles, DOIs and abstracts of the found papers
print("\n\nHere, we print the title, doi and abstract of research papers\n")
for paper in papers:
    print(f"{paper['title']}")
    print(f"{paper['doi']}")
    print(paper['abstract'])

# Get only these sentences containing numbers and a unit of GPa
import re
for paper in papers:
    paper['sent_num_gpa'] = []
    for sentence in paper['abstract']:
        if re.search(r'\d+', sentence):
            if re.search('GPa', sentence):
                paper['sent_num_gpa'].append(sentence)

for paper in papers:
    print(f"{paper['title']}")
    print(f"{paper['doi']}")
    print(paper['sent_num_gpa'])

# Let's classify using our previous classifier
for paper in papers:
    paper['is_bulk'] = []
    for sentence in paper['sent_num_gpa']:
        paper['is_bulk'].append(single_classifier(sentence, 'bulk modulus'))
        
# Let's check the results. Here, we "extract" data from research papers
# using freely available language models
print("\n\nHere, we start the extraction of the data from research articles\n")
for paper in papers:
    for i, classif in enumerate(paper['is_bulk']):
        if classif:
            print(paper['title'])
            print(paper['sent_num_gpa'][i])
            print()


###############################################################################

"""
 2. ChatGPT: Working with large Language Models (LLMs)

What is a typical large language model (LLM)?
A text completion model. It simply continues the provided input text
by adding the next most likely word (token, i.e., a few symbols, part of word,
short word, etc.)

In brief:
 - Input: text
 - Loop until the next predicted word is "END":
    1) Output: text + predicted next "word"
    2) Input = Output
 - Output: continuation of your text.


Trained on large amounts of text by hiding parts of it and trying
to predict them back.

ChatGPT:
Fine tuned to choose responses that seem like a conversation with another human


APIKey = ""

"""

#!pip install -q openai
import openai

### Put the API key provided by CIC
openai.api_key = ""

# A simple function to execute the API. Execute with the prompt
# (or conversation if there is some back-and-forth already),
# as strings in a table.
# The argument is a list containing the entire conversation,
# alternating between you and GPT

# Possible models: "gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-4o"
def prompt(QQ, typ='all', T=0.8, model="gpt-4o", syst=''):
    if typ == 'yn':
        tkn = 6
    elif typ == 'all':
        tkn = 1999
    elif typ == 'tab':
        tkn = 500
    Q = []
    roles = ['user', 'assistant']
    Q.append({'role': 'system', 'content': syst})
    for qqq in enumerate(QQ):
        Q.append({'role': roles[qqq[0] % 2], 'content': qqq[1]})
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=Q,
                temperature=T,
                max_tokens=tkn,
                frequency_penalty=0,
                presence_penalty=0
                )
            break
        except Exception as e:
            print("An error occurred:", e)
            if 'Please reduce the length of the messages' in str(e):
                print('TRUNCATING')
                if 'Use only data present in the text. If data is not present in the text, type' in Q[1]["content"]:
                    print(Q.pop(3))
                    print(Q.pop(3))
                else:
                    print(Q.pop(1))
                    print(Q.pop(1))
            elif 'per min' in str(e):
                print("Sleeping for 15 sec.")
                from time import sleep
                sleep(15)
    return(QQ, response['choices'][0]['message']['content'])


###############################################################################
# Example uses of ChatGPT in everyday life
# Important tip - all long form outputs from ChatGPT are for you to read,
# critically judge, iteratively improve, rewrite or both.
###############################################################################

# Example 1: Write an abstract based on bullet points
a = ['''Write an abstract for a seminar using the following points:
- ml workshop for undergraduate students
- chatgpt in general
- chatgpt for researchers
- chatgpt in materials research in particular
- main point chatgpt in materials data extraction: our latest paper - chatextract method''']
a, r = prompt(a)
print(r)

a.append(r)
a.append("Its on july 20th at 11am, so add that info. Also tell people to bring laptops.")
a, r = prompt(a)
print(r)


# Example 2: A journal with a 300 word limit for abstracts rejected you
# and the new journal wants 150 words.
abstr = "Modulated reflectance (contactless electroreflectance (CER), photoreflectance (PR), and piezoreflectance (PzR)) has been applied to study direct optical transitions in bulk MoS2, MoSe2, WS2, and WSe2. In order to interpret optical transitions observed in CER, PR, and PzR spectra, the electronic band structure for the four crystals has been calculated from the first principles within the density functional theory for various points of Brillouin zone including K and H points. It is clearly shown that the electronic band structure at H point of Brillouin zone is very symmetric and similar to the electronic band structure at K point, and therefore, direct optical transitions at H point should be expected in modulated reflectance spectra besides the direct optical transitions at the K point of Brillouin zone. This prediction is confirmed by experimental studies of the electronic band structure of MoS2, MoSe2, WS2, and WSe2 crystals by CER, PR, and PzR spectroscopy, i.e., techniques which are very sensitive to critical points of Brillouin zone. For the four crystals besides the A transition at K point, an AH transition at H point has been observed in CER, PR, and PzR spectra a few tens of meV above the A transition. The spectral difference between A and AH transition has been found to be in a very good agreement with theoretical predictions. The second transition at the H point of Brillouin zone (BH transition) overlaps spectrally with the B transition at K point because of small energy differences in the valence (conduction) band positions at H and K points. Therefore, an extra resonance which could be related to the BH transition is not resolved in modulated reflectance spectra at room temperature for the four crystals."
print(abstr, "\n")
abstr, r = prompt(["Make this research paper abstract no more than 150 words. Change only what is absolutely necessary, make as few changes as possible. " + abstr])
#abstr, r = prompt(["Make this 150 words: " + abstr])
print(r)

abstr.append(r)
abstr.append("Explain the logic behind what you removed and what you kept.")
abstr, r = prompt(abstr)
print(r)


# Example 3: Turn RIS to bibtex
ref = '''TY  - JOUR
AU  - Gupta, Tanishq
AU  - Zaki, Mohd
AU  - Krishnan, N. M. Anoop
AU  - Mausam
PY  - 2022
DA  - 2022/05/03
TI  - MatSciBERT: A materials domain language model for text mining and information extraction
JO  - npj Computational Matesdrials
SP  - 102
VL  - 8
IS  - 1
SN  - 2057-3960
UR  - https://doi.org/10.1038/s41524-022-00784-w
DO  - 10.1038/s41524-022-00784-w
ID  - Gupta2022'''

ref, r = prompt(["Turn this into bibtex, fix any mistakes you find: " + ref])
#ref,r = prompt(["Turn this into bibtex, dont write anything else, dont put the code into a code block: " + ref])
print(r)


###############################################################################

"""
 3. Use ChatGPT to classify text

Typical classification language models are trained to classify text,
and do only that:

Input: text

Output: 0-1 value for a given class
ChatGPT is trained to respond like a human:

Input: text

Output: The most likely response of a human to the content of input

"""

texts=[
"Transition metal dichalcogenides (TMDs), especially in two-dimensional (2D) form, exhibit many properties desirable for device applications. However, device performance can be hindered by the presence of defects. Here, we combine state of the art experimental and computational approaches to determine formation energies and charge transition levels of defects in bulk and 2D MX 2 (M = Mo or W; X = S, Se, or Te). We perform deep level transient spectroscopy (DLTS) measurements of bulk TMDs. Simultaneously, we calculate formation energies and defect levels of all native point defects, which enable identification of levels observed in DLTS and extend our calculations to vacancies in 2D TMDs, for which DLTS is challenging. We find that reduction of dimensionality of TMDs to 2D has a significant impact on defect properties. This finding may explain differences in optical properties of 2D TMDs synthesized with different methods and lays foundation for future developments of more efficient TMD-based devices.",
"We use a random forest (RF) model to predict the critical cooling rate (RC) for glass formation of various alloys from features of their constituent elements. The RF model was trained on a database that integrates multiple sources of direct and indirect RC data for metallic glasses to expand the directly measured RC database of less than 100 values to a training set of over 2000 values. The model error on 5-fold cross-validation (CV) is 0.66 orders of magnitude in K/s. The error on leave-out-one-group CV on alloy system groups is 0.59 log units in K/s when the target alloy constituents appear more than 500 times in training data. Using this model, we make predictions for the set of compositions with melt-spun glasses in the database and for the full set of quaternary alloys that have constituents which appear more than 500 times in training data. These predictions identify a number of potential new bulk metallic glass systems for future study, but the model is most useful for the identification of alloy systems likely to contain good glass formers rather than detailed discovery of bulk glass composition regions within known glassy systems.",
"The electronic band structure of MoS2, MoSe2, WS2, and WSe2, crystals has been studied at various hydrostatic pressures experimentally by photoreflectance (PR) spectroscopy and theoretically within the density functional theory (DFT). In the PR spectra direct optical transitions (A and B) have been clearly observed and pressure coefficients have been determined for these transitions to be: αA = 2.0 ± 0.1 and αB = 3.6 ± 0.1 meV/kbar for MoS2, αA = 2.3 ± 0.1 and αB = 4.0 ± 0.1 meV/kbar for MoSe2, αA = 2.6 ± 0.1 and αB = 4.1 ± 0.1 meV/kbar for WS2, αA = 3.4 ± 0.1 and αB = 5.0 ± 0.5 meV/kbar for WSe2. It has been found that these coefficients are in an excellent agreement with theoretical predictions. In addition, a comparative study of different computational DFT approaches has been performed and analyzed. For indirect gap the pressure coefficient have been determined"]

for i in texts:
    print(i)
    print()

print(single_classifier(texts[0],'machine learning'))
print(single_classifier(texts[1],'machine learning'))
print(single_classifier(texts[2],'machine learning'))

a, r = prompt(["What is the following text about?\n\n"+texts[1]])
print(r)

a, r = prompt(["What is the following text about? Answer with one to three words.\n\n"+texts[1]])
print(r)

a, r = prompt(["Is the following text about machine learning?\n\n"+texts[1]])
print(r)

a, r = prompt(["Is the following text about machine learning? Answer 'Yes' or 'No' only.\n\n"+texts[1]])
print(r)

if 'yes' in r.lower():
    print(True)
else:
    print(False)

classif = []
for t in texts:
    a, r = prompt(["Is the following text about machine learning? Answer 'Yes' or 'No' only. "+t])
    if 'yes' in r.lower():
        classif.append(True)
    else:
        classif.append(False)
print(classif)


# Let's think about this in terms of data extraction.
# We can classify text by whether it does or does not contain data we want.
# Imagine we are interested in extracting band gaps.
# Band gap is a property characteristic of semiconductors
# and insulators and is in units of energy, typically eV.

texts=[
"The sky is blue and the grass is green.",
"The exciton binding energy in GaAs in 0.03 eV.",
"Optical experiments revealed a band gap of 1.5 eV, with exciton binding energy of 30 meV."]

classif = []
for t in texts:
    a,r = prompt(["Does the following text contain the value of band gap? Answer 'Yes' or 'No' only. "+t])
    if 'yes' in r.lower():
        classif.append(True)
    else:
        classif.append(False)
print(classif)

a = "GaAs was studied next. Optical experiments revealed a band gap of 1.5 eV, with exciton binding energy of 30 meV."
a, r = prompt(["Summarize the data given in the following text: \n\n" + a])
print(r)

a = "GaAs was studied next. Optical experiments revealed a band gap of 1.5 eV, with exciton binding energy of 30 meV."
a, r = prompt(["Summarize ONLY the band gap data and nothing else, given the following text: \n\n" + a])
print(r)

a = "GaAs was studied next. Optical experiments revealed a band gap of 1.5 eV, with exciton binding energy of 30 meV."
a, r = prompt(["What is the material for which the band gap is given in the following text? \n\n" + a])
print(r)
a.append(r)
a, r = prompt(a+["What is the value of the band gap given in the following text? Provide only the value. \n\n" + a[0]])
print(r)
a.append(r)
a, r = prompt(a+["What is the unit of the band gap given in the following text? Provide only the unit. \n\n" + a[0]])
print(r)

a = "GaAs was studied next. Optical experiments revealed a band gap of 1.5 eV, with exciton binding energy of 30 meV."
a, r = prompt(["What is the material for which the band gap is given in the following text? Give the material name only, do not use a full sentence. \n\n" + a])
print(r)
a.append(r)
a, r = prompt(a + ["What is the value of the band gap given in the following text? Give the number only, do not use a full sentence. \n\n" + a[0]])
print(r)
a.append(r)
a, r = prompt(a + ["What is the unit of the band gap given in the following text? Give the unit only, do not use a full sentence. \n\n" + a[0]])
print(r)

t = "The band gaps for GaAs and GaSb were 1.5 and 0.8 eV, respectively, with exciton binding energy of around 0.03 eV for both, similar to that of GaN for which Eb=0.02 eV"
a, r = prompt(["Summarize the values of band gap in this text in a table consisting of 'Material, Value, Unit':\n\n" + t])
print(r)

a.append(r)
a, r = prompt(a + ["There is a possibility that the table you just extracted is incorrect. Is 1.5 the value of the band gap for the first material in the following text? Answer 'Yes' or 'No' only.\n\n" + t])
print(r)

a.append(r)
a, r = prompt(a + ["There is a possibility that the table you just extracted is incorrect. Is 0.02 the value of the band gap for the third material in the following text? Answer 'Yes' or 'No' only.\n\n\n\n" + t])
print(r)


###############################################################################

"""
 3. ChatExctract
 
Workflow that allows to extract information from scientific publications
The key aspects of ChatExtract behind its success:
- Redundancy
- Inducing uncertainty
- Allowing for negative answers
- Keeping the workflow in the form of a conversation, not separate questions

Check:
"Flexible, Model-Agnostic Method for Materials Data Extraction from Text 
 Using General Purpose Language Models"
Maciej P. Polak et al., Digital Discovery 2024, 3, 1221-1235
https://doi.org/10.1039/D4DD00016A

"""

# Specify the name of the property we are going to extract
PROPERTY = 'critical cooling rate'

# Download a csv file that contains sentences from papers and paper details.
# Normally you would download these papers yourself and then prepare the file.
# Simple but it takes time.

# !wget https://transfer.sh/Dft2z65vKX/chatextract_example_cc.csv
CSV_INPUT = 'chatextract_example_cc.csv'
import pandas as pd
test_df = pd.read_csv(CSV_INPUT)

print(test_df.keys())

test_df["passage"] = test_df["doi"]
for i in range(len(test_df["sentence"])):
    test_df["passage"][i] = test_df["title"][i] + " " + test_df["previous"][i]
    + " " + test_df["sentence"][i]

print(test_df.keys())
print()
for i in test_df["passage"]:
    print(i)
print()

for i in test_df["sentence"]:
    print(i)
    print()


# All the possible questions from the ChatExtract workflow
classif_q = 'Answer "Yes" or "No" only. Does the following text contain a value of '+PROPERTY+'?\n\n'
ifmulti_q = 'Answer "Yes" or "No" only. Does the following text contain more than one value of '+PROPERTY+'?\n\n'
single_q = [
'Give the number only without units, do not use a full sentence. If the value is not present in the text, type "None". What is the value of the '+PROPERTY+' in the following text?\n\n',
'Give the unit only, do not use a full sentence. If the unit is not present in the text, type "None". What is the unit of the '+PROPERTY+' in the following text?\n\n',
'Give the name of the material only, do not use a full sentence. If the name of the material is not present in the text, type "None". What is the material for which the '+PROPERTY+' is given in the following text?\n\n'
]

tab_q = 'Use only data present in the text. If data is not present in the text, type "None". Summarize the values of '+PROPERTY+' in the following text in a form of a table consisting of: Material, Value, Unit\n\n'
tabfollowup_q = [
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is "','" the ',' compound for which the value of '+PROPERTY+' is given in the following text? Make sure it is a real compound.\n\n'],
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ',' the value of the '+PROPERTY+' for the ',' compound in the following text?\n\n'],
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ',' the unit of the ',' value of '+PROPERTY+' in the following text?\n\n']
]

# This is just the ChatExtract workflow written up as a code. No need to worry
# about it unless you want to play around with it and change stuff. It will go
# through the workflow, ask questions, and take actions based on the responses.

from re import split
it = [ 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
      'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth',
      'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth',
      'nineteenth', 'twentieth']
col = ['Material', 'Value','Unit']

single_cols = ['value', 'unit', 'material']

extracted = ["passage", "doi", "material", "value", "unit", "material_valid",
             "value_valid", "unit_valid"]
extracted = []

binclas_final = []

ntot=len(test_df)
for i in range(len(test_df)):
    try:
        binary_classif = 0
        sss=[]
        print("Processing ", CSV_INPUT, " ", i, " ", round(i/ntot*100, 1), "%")
        ss = classif_q + test_df["sentence"][i]
        sss.append(ss)
        sss, ans = prompt(sss, 'yn')
        sss.append(ans)
        if 'yes' in ans.strip().lower():
            binary_classif = 1
            result = {}
            ss = ifmulti_q+test_df["passage"][i]
            sss.append(ss)
            sss, ans = prompt(sss, 'yn')
            sss.append(ans)
            if 'no' in ans.lower():
                result["passage"] = [test_df["passage"][i]]
                result["doi"] = [test_df["doi"][i]]
                result["material"] =[]
                result["value"] =[]
                result["unit"] =[]
                result["material_valid"] =[]
                result["value_valid"] =[]
                result["unit_valid"] =[]
                for j in range(len(single_q)):
                    ss = single_q[j]+test_df["passage"][i]
                    sss.append(ss)
                    sss,ans = prompt(sss, 'all')
                    sss.append(ans)
                    result[single_cols[j]].append(ans)
                    if 'none' in ans.lower():
                        result[single_cols[j]+"_valid"].append(0)
                    else:
                        result[single_cols[j] + "_valid"].append(1)
            elif 'yes' in ans.lower():
                ss = tab_q + test_df["passage"][i]
                sss.append(ss)
                sss, tab = prompt(sss, 'tab')
                sss.append(tab)
                tab = [split('[,|]', row) for row in tab.strip().split('\n')]
                tab = [[item.strip() for item in row if len(item.strip())>0] for row in tab if len(row)>=3]
                if len(tab)<=0:
                    tab.append(['Material', 'Value', 'Unit'])
                if len(tab)<=1:
                    tab.append(['None', 'None', 'None'])
                else:
                    tab.pop(1)
                head = tab.pop(0)
                tab = pd.DataFrame(tab, columns=head)
                result["passage"] = []
                result["doi"] = []
                result["material"] = []
                result["value"] = []
                result["unit"] = []
                result["material_valid"] = []
                result["value_valid"] = []
                result["unit_valid"] = []
                for k in range(len(tab)):
                    result["passage"].append(test_df["passage"][i])
                    result["doi"].append(test_df["doi"][i])
                    multi_valid = True
                    for l in range(3):
                        ss = tabfollowup_q[l][0] + str(tab[col[l]][k])
                        + tabfollowup_q[l][1]
                        + it[k]
                        + tabfollowup_q[l][2]
                        + test_df["passage"][i]
                        result[col[l].lower()].append(tab[col[l]][k])
                        if 'none' in tab[col[l]][k].lower():
                            result[col[l].lower() + "_valid"].append(0)
                            multi_valid = False
                        elif multi_valid:
                            sss.append(ss)
                            sss,ans = prompt(sss, 'yn')
                            sss.append(ans)
                            if 'no' in ans.lower():
                                result[col[l].lower() + "_valid"].append(0)
                                multi_valid = False
                            else:
                                result[col[l].lower() + "_valid"].append(1)
                        else:
                            result[col[l].lower() + "_valid"].append(1)
            extracted.append(result)
        else:
            binary_classif = 0
    except:
        print('Something went wrong')
    binclas_final.append(binary_classif)


# Let's look at the results
from pprint import pprint
for i in extracted:
    if 'None' not in i['value']:
      pprint(i)

# Databases we developed so far:
# 1) Critical Cooling rates for metallic glasses - 125 final points
#    with uniquely identifiable compositions (previous <100 points)
# 2) Yield strengths of High Entropy Alloys - 645 final points
#    with uniquely identifiable compositions (previous ~300 points)


###############################################################################

""" EXERCISE: Classify sentences from abstracts for a different property

Maybe try different classification models from huggingface on bulk modulus
and see the difference? Explore, ask questions.

Below is a copy of the classification workflow we started with, for your convenience.


# Search for papers
papers = search_arxiv(['critical cooling rate'])

# Printing the titles, DOIs and abstracts of the found papers
for paper in papers:
  print(f"{paper['title']}")
  print(f"{paper['doi']}")
  print(paper['abstract'])

# Reduce to sentences with numbers (and maybe some unit?)
import re
for paper in papers:
  paper['sent_num']=[]
  for sentence in paper['abstract']:
    if re.search(r'\d+', sentence):
      if re.search('K', sentence):   #Unit, I used Kelvin here - for temperature
        paper['sent_num'].append(sentence)

# See what's left
for paper in papers:
    print(f"{paper['title']}")
    print(f"{paper['doi']}")
    print(paper['sent_num'])

# Classify
for paper in papers:
  paper['is_bulk']=[]
  for sentence in paper['sent_num']:
    paper['is_bulk'].append(single_classifier(sentence,'critical cooling rate'))

# Analyze results
for paper in papers:
  for i,classif in enumerate(paper['is_bulk']):
    if classif:
      print(paper['title'])
      print(paper['sent_num'][i])
      print()
      
"""