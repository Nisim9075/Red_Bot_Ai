from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import time
import joblib

class MachineLearningModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()  # השתמש ב-TfidfVectorizer
        self.model = MultinomialNB()
        self.is_trained = False  # משתנה לבדיקת האם המודל אומן
        self.data = []  # רשימת השאלות
        self.labels = []  # רשימת התשובות
        self.training_time = 0  # משתנה לאחסון זמן האימון

    def train(self, data, labels):
        """אימון המודל עם הנתונים הקיימים"""
        start_time = time.time()  # התחלת ספירת הזמן לאימון
        X = self.vectorizer.fit_transform(data)
        self.model.fit(X, labels)
        self.is_trained = True  # סימון שהמודל אומן
        self.data = data
        self.labels = labels
        self.training_time = time.time() - start_time  # סיום ספירת הזמן
        print(f"Training completed in {self.training_time:.2f} seconds.")

    def predict(self, text):
        """חיזוי תשובה למילה או שאלה"""
        if not self.is_trained:
            raise ValueError("❌ שגיאה: המודל לא אומן! יש לקרוא ל- train() לפני השימוש.")
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def predict_with_confidence(self, text):
        """חיזוי תשובה עם רמת בטחון"""
        if not self.is_trained:
            raise ValueError("❌ שגיאה: המודל לא אומן! יש לקרוא ל- train() לפני השימוש.")
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)
        prediction = self.model.predict(X)[0]
        confidence = np.max(proba)  # רמת הבטחון
        return prediction, confidence

    def add_new_data(self, new_data, new_label):
        """הוספת נתונים חדשים למודל"""
        self.data.append(new_data)  # הוספת השאלה
        self.labels.append(new_label)  # הוספת התשובה
        # לא תמיד חייבים לאמן מחדש כל פעם, אפשר לבחור אם רוצים לאמן מחדש
        self._retrain_if_needed()

    def add_bulk_data(self, new_data, new_labels):
        """הוספת כמות גדולה של נתונים בצורה יעילה"""
        self.data.extend(new_data)  # הוספת רשימת השאלות
        self.labels.extend(new_labels)  # הוספת רשימת התשובות
        # לא תמיד חייבים לאמן מחדש כל פעם, אפשר לבחור אם רוצים לאמן מחדש
        self._retrain_if_needed()

    def _retrain_if_needed(self):
        """שיפור הוספת נתונים כך שלא תמיד צריך לאמן מחדש"""
        if len(self.data) % 20 == 0:  # לדוגמה, נעדכן את המודל כל 20 נתונים חדשים
            print("Training model with new data...")
            self.train(self.data, self.labels)

    def save_model(self, filename):
        """שמור את המודל והווקטוריזר לקובץ"""
        joblib.dump(self.model, filename + "_model.pkl")
        joblib.dump(self.vectorizer, filename + "_vectorizer.pkl")
        print(f"Model saved to {filename}_model.pkl and {filename}_vectorizer.pkl")

    def load_model(self, filename):
        """טען את המודל והווקטוריזר מקובץ"""
        self.model = joblib.load(filename + "_model.pkl")
        self.vectorizer = joblib.load(filename + "_vectorizer.pkl")
        self.is_trained = True
        print(f"Model loaded from {filename}_model.pkl and {filename}_vectorizer.pkl")


# יצירת מופע של המודל
ml_model = MachineLearningModel()

# אימון ראשוני משודרג למניעת שגיאות
sample_data = [
    "שלום", "מה שלומך?", "איך קוראים לך?", "ביי", "מה נשמע?", "מה חדש?", "איך אתה?", "מה אתה עושה?", 
    "אתה יכול לעזור לי?", "מה אתה יודע?"
]

sample_labels = [
    "היי!", "אני בסדר, תודה!", "אני צ'אטבוט!", "להתראות!", "הכל טוב, איך אתה?", "לא הרבה, מה איתך?", 
    "אני כאן כדי לעזור!", "אני לא עושה הרבה, רק עוזר לך!", "כמובן! איך אני יכול לעזור?", 
    "אני יודע הרבה דברים! תוכל לשאול אותי כל שאלה."
]

# אימון המודל
ml_model.train(sample_data, sample_labels)

# הוספת שאלה ותשובה חדשה למודל
ml_model.add_new_data("מה שלומך?", "אני מצוין, תודה ששאלת!")  # הוספת שאלה חדשה
ml_model.add_new_data("מה עשית היום?", "היום אני עוזר למשתמשים כמוך! איך היה היום שלך?")  # שאלה על פעילות המשתמש

# הוספת רשימת שאלות ותשובות בצורה יעילה
bulk_data = [
    "מה עושה אותך מאושר?", "מה זה בית בשבילך?", "יש לך חברים קרובים?", "מהו חלום שלך?", 
    "מה היית רוצה ללמוד?", "מהו הספר האהוב עליך?", "מהו הדבר הכי חשוב בעיניך?", "מה דעתך על מוזיקה?", 
    "מה עושה אותך שמח?", "מה דעתך על טכנולוגיה?"
]

bulk_labels = [
    "מאושר הוא מילה גדולה. מה עושה אותך מאושר?", "הבית הוא המקום שבו אתה מרגיש בבית. איך אתה רואה את הבית שלך?",
    "חברים קרובים חשובים בחיים. איך אתה רואה את הקשרים החברתיים שלך?", "חלום הוא משהו שמניע אותנו קדימה. איזה חלום יש לך?",
    "העולם מלא בדברים ללמוד. יש משהו שמעניין אותך ללמוד?", "ספרים יכולים לשנות חיים. יש ספר שמאוד השפיע עליך?",
    "הדברים החשובים משתנים אצל כל אדם. מה הדבר הכי חשוב לך בחיים?", "מוזיקה היא חלק מהחיים! יש לך סוג מוזיקה אהוב?",
    "הדברים הקטנים יכולים להיות המרגשים ביותר. מה עושה אותך שמח?", "טכנולוגיה היא העתיד! איך אתה רואה את השפעתה על חיינו?"
]

# הוספת כמות גדולה של נתונים למודל בצורה יעילה
ml_model.add_bulk_data(bulk_data, bulk_labels)

# שמירת המודל
ml_model.save_model("chatbot_model")

# טעינת המודל
ml_model.load_model("chatbot_model")

# חיזוי עם רמת בטחון
prediction, confidence = ml_model.predict_with_confidence("מה שלומך?")
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
