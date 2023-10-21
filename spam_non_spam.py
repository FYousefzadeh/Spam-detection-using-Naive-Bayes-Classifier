import os
import random
import math
import re


class NaiveBayesClassifier:
    def __init__(self, vocab):
        self.word_counts = {'spam': {}, 'nospam': {}}
        self.class_counts = {'spam': 0, 'nospam': 0}
        self.total_email_count = 0
        self.vocab = vocab

    def train(self, emails, labels):
        for email, label in zip(emails, labels):
            self.process_email(email, label)

        self.total_email_count = len(emails)

    def process_email(self, email, label):
        words = self.preprocess_email(email).split()
        for word in words:
            if word not in self.vocab:
                continue

            word_counts = self.word_counts[label]
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

        self.class_counts[label] += 1

    def classify(self, email):
        words = email.split()
        spam_probability = self.calculate_probability(words, 'spam')
        nospam_probability = self.calculate_probability(words, 'nospam')

        if spam_probability > nospam_probability:
            return 'spam'
        else:
            return 'nospam'

    def calculate_probability(self, words, label):
        word_counts = self.word_counts[label]
        class_count = self.class_counts[label]

        total_word_count = sum(word_counts.values())
        probability = math.log(class_count / self.total_email_count)

        for word in words:
            if word in self.vocab and word in word_counts:
                word_occurrences = word_counts[word]
                probability += math.log((word_occurrences + 1) / (total_word_count + len(self.vocab)))

        return probability

    @staticmethod
    def preprocess_email(email):
        # removing numbers
        email = re.sub(r'\d+(\.\d+)?', 'number', email)
        # expanding contractions
        email = re.sub(r"won't", "will not", email)
        email = re.sub(r"can\'t", "can not", email)
        email = re.sub(r"n\'t", " not", email)
        email = re.sub(r"\'re", " are", email)
        email = re.sub(r"\'s", " is", email)
        email = re.sub(r"\'d", " would", email)
        email = re.sub(r"\'ll", " will", email)
        email = re.sub(r"\'t", " not", email)
        email = re.sub(r"\'ve", " have", email)
        email = re.sub(r"\'m", " am", email)
        # changing the mail to lowercase
        email = email.lower()
        # replacing next lines by "white space"
        email = email.replace(r'\n', " ")
        # replacing email IDs by "MailID"
        email = email.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'MailID')
        # replacing URLs  by "Links"
        email = email.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'Links')
        # replacing currency signs by "Money"
        email = email.replace(r'£|\$', 'Money')
        # replacing "special character"  by "white space"
        email = email.replace(r"[^a-zA-Z0-9]+", " ")
        # replacing all kinds of space by "single white space"
        email = email.replace(r'\s+', ' ')
        email = email.replace(r'^\s+|\s+?$', '')
        # replacing "contact numbers" by "contact number"
        email = email.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'contact number')
        return email


def load_lingspam_dataset(directory):
    emails = []
    labels = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                email = f.read()
                emails.append(email)
                if file.startswith('spm'):
                    labels.append('spam')
                else:
                    labels.append('nospam')

    return emails, labels


def split_dataset(emails, labels, train_ratio):
    combined = list(zip(emails, labels))
    random.shuffle(combined)
    emails_shuffled, labels_shuffled = zip(*combined)
    split_index = int(len(emails) * train_ratio)
    train_emails = emails_shuffled[:split_index]
    train_labels = labels_shuffled[:split_index]
    test_emails = emails_shuffled[split_index:]
    test_labels = labels_shuffled[split_index:]

    return train_emails, train_labels, test_emails, test_labels


def calculate_accuracy(predictions, labels):
    correct_count = sum(1 for i in range(len(predictions)) if predictions[i] == labels[i])
    accuracy = correct_count / len(predictions)
    return accuracy

# loading and preprocessing the data
lingspam_directory = r"\dataset_path"
emails, labels = load_lingspam_dataset(lingspam_directory)
vocab = set()
for email in emails:
    words = NaiveBayesClassifier.preprocess_email(email).split()
    vocab.update(words)

# Split the dataset into train and test datasets and training the classifier
train_emails, train_labels, test_emails, test_labels = split_dataset(emails, labels, train_ratio=0.8)
classifier = NaiveBayesClassifier(vocab)
classifier.train(train_emails, train_labels)

# Classify the test data
test_predictions = [classifier.classify(email) for email in test_emails]

# Calculate accuracies
train_accuracy = calculate_accuracy([classifier.classify(email) for email in train_emails], train_labels)
test_accuracy = calculate_accuracy(test_predictions, test_labels)
print("Training accuracy:", round(train_accuracy * 100, 2))
print("Testing accuracy:", round(test_accuracy * 100, 2))
