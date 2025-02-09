from flask import Flask, render_template, request
import pandas as pd
from fuzzywuzzy import process
from nltk import word_tokenize, pos_tag
import nltk

app = Flask(__name__)

# Load the CSV file containing all tables
data_df = pd.read_csv('placenamedatasets.csv')

# Data Preprocessing
def preprocess_data(data_df):
    # Standardize capitalization
    data_df = data_df.apply(lambda x: x.str.capitalize() if x.dtype == 'object' else x)
    # Remove duplicates
    data_df = data_df.drop_duplicates().reset_index(drop=True)
    return data_df

data_df = preprocess_data(data_df)

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def identify_place_names(sentence):
    tokens = word_tokenize(sentence)
    tokens = [token.capitalize() for token in tokens]
    tagged_tokens = pos_tag(tokens)
    identified_places = []

    # Initialize the highlighted sentence
    highlighted_sentence = sentence

    for token, pos in tagged_tokens:
        if pos == 'NNP':
            matches = [
                process.extractOne(token.lower(), data_df[table].str.lower())
                for table in ['Country', 'State', 'City']
            ]

            # Check if the token has a space and try to match it as a multi-word place name
            if ' ' in token:
                matches.append((token.lower(), 100, 'Multi-word'))

            best_match, score, table = max(matches, key=lambda x: x[1])

            # Highlight the identified place names in the sentence
            if score >= 92:
                if table != 'Multi-word':
                    highlighted_name = f'<span style="color: green">{token}</span>' if score == 100 else f'<span style="color: red">{token}</span>'
                    highlighted_sentence = highlighted_sentence.replace(token, highlighted_name)

                if table == 'Country':
                    place_type = 'Country'
                elif table == 'State':
                    place_type = 'State'
                elif table == 'City':
                    place_type = 'City'
                else:
                    place_type = 'Multi-word'

                identified_places.append(
                    {'Token': token, 'Canonical name': best_match.title(), 'Place Type': place_type, 'Score': score})

    return highlighted_sentence, identified_places



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            try:
                # Read the uploaded file and extract text content with latin-1 encoding
                content = file.read().decode('latin-1')
                highlighted_sentence, identified_places = identify_place_names(content)
                return render_template('result.html', highlighted_sentence=highlighted_sentence, identified_places=identified_places)
            except Exception as e:
                return render_template('error.html', error_message=str(e))

    # If no file uploaded, fallback to identifying place names from the entered sentence
    sentence = request.form['sentence']
    highlighted_sentence, identified_places = identify_place_names(sentence)
    return render_template('result.html', highlighted_sentence=highlighted_sentence, identified_places=identified_places)


if __name__ == '__main__':
    app.run(debug=True)

