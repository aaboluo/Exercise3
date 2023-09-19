# coding: utf-8
import nltk
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


def write_file(filename, data):
    with open(filename, "w") as f:
        f.write(str(data))


# init
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Reads the Moby Dick file from the Gutenberg dataset
nltk.corpus.gutenberg.fileids()
moby_dick = gutenberg.raw("melville-moby_dick.txt")

# 1. Tokenization Moby Dick
tokens = nltk.word_tokenize(moby_dick)
write_file("1.txt", tokens)

# 2. Stop-words filtering: Filters out the stopwords from the above tokens.
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
write_file("2.txt", filtered_tokens)

# 3. POS frequency: The program then counts and displays ... and their total counts (frequency).
word_tags = pos_tag(filtered_tokens)
write_file("3.txt", word_tags)

# 4. POS frequency: The program then counts and displays the 5 most ... and their total counts (frequency).
pos_dict = {}
punctuations = string.punctuation
for word, pos in word_tags:
    if word[0] in punctuations:
        continue
    if pos not in pos_dict:
        pos_dict[pos] = 0
    pos_dict[pos] += 1

sorted_pos_tup = sorted(pos_dict.items(), key=lambda x: x[1], reverse=True)
pos_string = ""
for pos, frequency in sorted_pos_tup[:5]:
    pos_string += f"{pos}: {frequency}\n"
write_file("4.txt", pos_string)

# 5. Lemmatization: Using the pos-tagged tokens, ... the root of “singing”, “singer”, “sings”, “sang”, and “sung” is “sing”.
lemmatizer = WordNetLemmatizer()
top_20_tokens = [token for token, _ in word_tags[:20]]
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in top_20_tokens]
write_file("5.txt", lemmatized_tokens)

# 6. Plotting frequency distribution: At the end, ... and their total occurrences are plotted as a bar chart.
datearr = list(pos_dict.keys())
num_masks = list(pos_dict.values())
fig = plt.figure(figsize=(15, 6))
plt.xlabel('pos', fontsize=10)
plt.ylabel('frequency', fontsize=10)
plt.title('6.Plotting frequency distribution', fontsize=15)
plt.bar(datearr, num_masks, width=0.3)
plt.savefig('6.jpg', dpi=1000)
plt.show()
