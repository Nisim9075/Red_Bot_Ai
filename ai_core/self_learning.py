from database.database import save_interaction
from models.machine_learning import ml_model
from datetime import datetime
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob  # לשימוש בהערכת רגש
import hashlib
import langid  # לזיהוי אוטומטי של שפה
import joblib  # לשמירת מודלים
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from langdetect import detect  # זיהוי שפות נוספות

class SelfLearning:
    def __init__(self):
        self.new_data = []
        self.retrain_frequency = 100  # מספר אינטראקציות לפני כל אימון מחדש
        self.stop_words_english = set(stopwords.words('english'))  # מילים מיותרות באנגלית
        self.stop_words_hebrew = set(stopwords.words('hebrew'))  # מילים מיותרות בעברית
        self.known_interactions = set()  # סט לשמירת אינטראקציות שנלמדו, לשם מניעת כפילויות
        self.stats = {"total_interactions": 0, "retrain_count": 0}  # סטטיסטיקות על האינטראקציות
        self.vectorizer = TfidfVectorizer(max_features=5000)  # משתנה ל-Tfidf Vectorizer
        self.model = None  # יש לאתחל את המודל בעת האימון

    def clean_text(self, text, language='english'):
        """פונקציה לניקוי טקסט, להסיר סימנים, מילים מיותרות, וכו'"""
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = word_tokenize(text)
        
        if language == 'english':
            stop_words = self.stop_words_english
        elif language == 'hebrew':
            stop_words = self.stop_words_hebrew
        else:
            stop_words = set()

        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    def check_sentiment(self, text):
        """פונקציה להערכת רגש בתשובה של המשתמש"""
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        return sentiment >= 0.2

    def detect_language(self, text):
        """פונקציה לזיהוי אוטומטי של שפה"""
        try:
            lang = detect(text)
            return lang
        except:
            return None

    def validate_data(self, user_input, bot_response):
        """פונקציה לבדוק אם הנתונים תקינים לפני שהם מאומנים"""
        user_lang = self.detect_language(user_input)
        bot_lang = self.detect_language(bot_response)
        
        if user_lang != bot_lang:
            print(f"Invalid data: Mismatched languages. User input language: {user_lang}, Bot response language: {bot_lang}")
            return False
        
        if user_lang is None or bot_lang is None:
            print(f"Invalid data: Unsupported language detected.")
            return False
        
        user_input = self.clean_text(user_input, user_lang)
        bot_response = self.clean_text(bot_response, bot_lang)

        if len(user_input) < 5 or len(bot_response) < 5:
            print("Invalid data: Input or response is too short.")
            return False

        interaction_hash = hashlib.md5(f"{user_input}{bot_response}".encode()).hexdigest()
        if interaction_hash in self.known_interactions:
            print("Duplicate data: This interaction has already been learned.")
            return False

        if not self.check_sentiment(bot_response):
            print("Invalid data: Negative sentiment detected in response.")
            return False

        return True

    def learn_from_user(self, user_input, bot_response):
        """ללמוד מהמשתמש ולהוסיף את הנתונים"""
        if self.validate_data(user_input, bot_response):
            self.new_data.append((user_input, bot_response))
            save_interaction(user_input, bot_response)
            interaction_hash = hashlib.md5(f"{user_input}{bot_response}".encode()).hexdigest()
            self.known_interactions.add(interaction_hash)
            self.stats["total_interactions"] += 1
            
            if len(self.new_data) >= self.retrain_frequency:
                self.retrain_model()

    def retrain_model(self):
        """לאמן את המודל עם הנתונים החדשים"""
        if self.new_data:
            data, labels = zip(*self.new_data)
            print(f"Training model with {len(data)} new interactions...")
            
            # המרת הנתונים למאפיינים נומריים באמצעות TfidfVectorizer
            X = self.vectorizer.fit_transform(list(data))
            y = list(labels)

            # אלגוריתמים אפשריים
            models = {
                "logistic_regression": LogisticRegression(),
                "random_forest": RandomForestClassifier(n_estimators=100),
                "svm": SVC(),
                "naive_bayes": MultinomialNB(),
                "knn": KNeighborsClassifier()
            }

            best_model, best_score = None, 0

            # אופטימיזציה של hyperparameters
            param_grid = {
                'logistic_regression': {'C': [0.1, 1, 10]},
                'random_forest': {'n_estimators': [50, 100, 200]},
                'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
                'naive_bayes': {'alpha': [0.1, 1, 10]},
                'knn': {'n_neighbors': [3, 5, 7, 10]}
            }

            for model_name, model in models.items():
                print(f"Training {model_name}...")
                grid_search = GridSearchCV(model, param_grid[model_name], cv=5)
                grid_search.fit(X, y)

                best_model_for_this = grid_search.best_estimator_
                predictions = best_model_for_this.predict(X)
                score = accuracy_score(y, predictions)
                print(f"{model_name} accuracy: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = best_model_for_this

            self.model = best_model  # נשמור את המודל הטוב ביותר
            self.new_data = []  # ננקה את הנתונים אחרי האימון
            self.stats["retrain_count"] += 1
            print("Model retrained successfully.")

            # שמירת המודל בעזרת joblib
            joblib.dump(self.model, "best_model.pkl")
            joblib.dump(self.vectorizer, "vectorizer.pkl")

    def load_model(self):
        """לטעון מודל קיים אם יש"""
        try:
            self.model = joblib.load("best_model.pkl")
            self.vectorizer = joblib.load("vectorizer.pkl")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def print_stats(self):
        """הדפסת סטטיסטיקות על האינטראקציות והאימונים"""
        print(f"Total interactions: {self.stats['total_interactions']}")
        print(f"Total model retrains: {self.stats['retrain_count']}")

self_learning = SelfLearning()

# טעינת המודל אם קיים
self_learning.load_model()
