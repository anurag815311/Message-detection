# importing required libraries
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import warnings
import re
warnings.filterwarnings("ignore")

# Ensure required NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# reading the dataset
file_path = r"C:\Users\Dell\OneDrive\Desktop\Message-detection\Cleaned_Dataset.csv"  
msg = pd.read_csv(file_path)
msg.drop(['Unnamed: 0'], axis=1, inplace=True)

# separating target and features
y = pd.DataFrame(msg['label'])  # Ensure column 'label' exists
x = msg.drop(['label'], axis=1)

# CountVectorization
cv = CountVectorizer(max_features=5000)
temp1 = cv.fit_transform(x['final_text'].values.astype('U')).toarray()
tf = TfidfTransformer()
temp1 = tf.fit_transform(temp1)
temp1 = pd.DataFrame(temp1.toarray(), index=x.index)
x = pd.concat([x, temp1], axis=1, sort=False)

# drop final_text column
x.drop(['final_text'], axis=1, inplace=True)

# converting labels to int datatype
y = y['label'].astype(int)  # Extracting the series before converting

# RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x, y)

# User input
text = input("Enter text: ")

# Data cleaning/preprocessing - removing punctuation and digits
updated_text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])

# Data cleaning/preprocessing - tokenization and convert to lower case
text = re.split(r"\W+", updated_text.lower())

# Data cleaning/preprocessing - stopwords removal
stopwords = nltk.corpus.stopwords.words('english')
text = [word for word in text if word and word not in stopwords]

# Data cleaning/preprocessing - lemmatizing
wordlem = nltk.WordNetLemmatizer()
text = [wordlem.lemmatize(word) for word in text]

# Data cleaning/preprocessing - merging tokens
text = " ".join(text)

# Transforming the input text to match the trained model's format
text_vectorized = cv.transform([text])
text_transformed = tf.transform(text_vectorized)

# Prediction
pred = model.predict(text_transformed)
if pred[0] == 0:
    print("Not Spam")
else:
    print("Spam")
