import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from transformers import pipeline

# Load dataset
df = pd.read_csv("/Users/joaovasco/Downloads/archive-3/Reviews.csv")
df['HelpfulnessNumerator'] = pd.to_numeric(df['HelpfulnessNumerator'], errors='coerce')
df['HelpfulnessDenominator'] = pd.to_numeric(df['HelpfulnessDenominator'], errors='coerce')
df = df.dropna(subset=['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text'])
df['Helpful'] = df.apply(lambda x: x['HelpfulnessNumerator'] > 3, axis=1)

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Helpful'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Transformer Model for Sentiment Analysis
transformer_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy:.2f}")

def predict_helpfulness(comment):
    comment_vec = vectorizer.transform([comment])
    nb_prediction = models['Naive Bayes'].predict_proba(comment_vec)[:, 1][0]
    lr_prediction = models['Logistic Regression'].predict_proba(comment_vec)[:, 1][0]
    transformer_result = transformer_model(comment)
    transformer_pred = 1 if transformer_result[0]['label'] == 'POSITIVE' else 0

    # Assuming you have accuracy values for each model
    nb_weight = 0.85  # Replace with actual Naive Bayes accuracy
    lr_weight = 0.89  # Replace with actual Logistic Regression accuracy
    transformer_weight = 1.0  # Assign a weight to the transformer model

    total_weight = nb_weight + lr_weight + transformer_weight
    weighted_sum = (nb_prediction * nb_weight + lr_prediction * lr_weight + transformer_pred * transformer_weight)
    
    return weighted_sum / total_weight



# Existing code to predict helpfulness...

# Test with a custom comment
custom_comment = "This organic honey is not only delicious but also comes in eco-friendly packaging. I found it perfect for sweetening my tea and baking. The texture is smooth, and it has a rich, natural flavor that isn't too overpowering. Plus, it's great to know that it's sourced sustainably. Highly recommend for those who love natural sweeteners!"
likelihood = predict_helpfulness(custom_comment)

# Pretty output
print("\nReview Analysis Report")
print("----------------------")
print(f"Review Text: '{custom_comment}'")
print(f"Predicted Helpfulness Likelihood: {likelihood:.2f} (where 1.0 is highly likely and 0.0 is not likely)")
