import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

intents = {
    "greeting": [
        "hi", "hello", "hey", "good morning"
    ],
    "hours": [
        "what are your working hours",
        "when are you open",
        "business hours"
    ],
    "contact": [
        "how can i contact you",
        "customer support number",
        "email support"
    ],
    "returns": [
        "return policy",
        "how to return a product",
        "refund process"
    ],
    "order": [
        "where is my order",
        "track my order",
        "order status"
    ],
    "shipping": [
        "shipping details",
        "delivery information",
        "do you offer free delivery"
    ],
    "payment": [
        "payment methods",
        "how can i pay",
        "payment options available"
    ],
    "cancel": [
        "cancel my order",
        "order cancellation",
        "how to cancel order"
    ],
    "availability": [
        "is this product available",
        "check product stock",
        "product availability"
    ],
    "goodbye": [
        "bye", "thank you", "thanks"
    ]
}

responses = {
    "greeting": ["Hello! How can I help you?"],
    "hours": ["We are open from 9 AM to 6 PM, Monday to Friday."],
    "contact": ["You can contact us at support@company.com"],
    "returns": ["You can return products within 30 days."],
    "order": ["Please share your order ID to track it."],
    "shipping": ["We offer free shipping on orders above ₹500."],
    "payment": ["We accept UPI, cards, and net banking."],
    "cancel": ["Orders can be cancelled within 24 hours."],
    "availability": ["Please tell me the product name."],
    "goodbye": ["Goodbye! Have a great day 😊"],
    "default": ["Sorry, I couldn't understand that."]
}

sentences = []
labels = []

for intent, examples in intents.items():
    for example in examples:
        sentences.append(example)
        labels.append(intent)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    max_sim = similarity.max()

    if max_sim < 0.35:
        return None  

    intent = labels[similarity.argmax()]
    return random.choice(responses[intent])
