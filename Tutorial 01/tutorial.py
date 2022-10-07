# Write a program that reads in a text file and prints out the word frequencies. Try to use Jupiter lab notebook to document the development of your program. Example
# Input text: “The rain fell on the car. The rain fell on the ground.”
# Output: the-3, rain-2, fell-2, on-2, car-1, ground-1

# Importing the required libraries
import re
import string
import operator

# Defining the function to read the file
def read_file():
    with open('data.txt', 'r') as f:
        data = f.read().replace('\n', '')
        return data

# Defining the function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Defining the function to count the words
def count_words(text):
    word_list = text.split()
    word_count = {}
    for word in word_list:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

# Defining the function to sort the words
def sort_words(word_count):
    sorted_words = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_words

# Defining the function to print the words
def print_words(sorted_words):
    for word in sorted_words:
        print(word[0], word[1])

# Calling the functions
text = read_file()
clean_text = clean_text(text)
word_count = count_words(clean_text)
sorted_words = sort_words(word_count)
print_words(sorted_words)

