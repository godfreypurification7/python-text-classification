from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

text=[
    "I like this",
    "This is amazing",
    "I really like it",
    "This is great",
    "I hate this",
    "This is awful",
    "Ireally dislike it",
    "This is terrible"
        ]

labels = ["positive","positive","positive","positive",
        "negative","negative","negative","negative"]

vectorzier=TfidfVectorizer()
numeric_data=vectorzier.fit_transform(text)

model=MultinomialNB()
model.fit(numeric_data,labels)

def predict(text):
    x=vectorzier.transform([text])
    return model.predict(x)[0]
print(predict("I love this video"))
print(predict("You are a bad person"))
print(predict("I like this product but it has problem"))
print(predict("I like this product and it is great"))
print(predict("He has been sentenced for 10 years in prison"))
print(predict("I always agree to divorce"))
print(predict("it is painful but I like divorcie."))