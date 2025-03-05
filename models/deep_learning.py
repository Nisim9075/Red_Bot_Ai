import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class DeepLearningModel:
    def __init__(self, vocab_size=10000, embedding_dim=16, hidden_units=16):
        # בניית המודל עם שכבות גמישות
        self.model = keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_dim),  # שכבת Embedding לגיבוב מילים
            keras.layers.GlobalAveragePooling1D(),  # Pooling של אורך הטקסט
            keras.layers.Dense(hidden_units, activation='relu'),  # שכבת Dense עם פונקציית ReLU
            keras.layers.Dense(1, activation='sigmoid')  # שכבת יציאה עם פונקציית Sigmoid להתאמת קטגוריה בינארית
        ])
        # קומפילציה של המודל
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        # תהליך האימון כולל הגדרת אופטימיזציה אישית לשיעור למידה
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, X):
        # תהליך החיזוי של המודל
        return self.model.predict(np.array([X]))[0]

    def evaluate(self, X_test, y_test):
        # פונקציה להערכת המודל על סט בדיקה
        return self.model.evaluate(X_test, y_test)

    def summary(self):
        # פונקציה להדפסת סיכום המודל
        self.model.summary()

# דוגמת נתונים - טקסטים ותוויות
sample_data = [
    "שלום", "מה שלומך?", "איך קוראים לך?", "ביי", "מה נשמע?", 
    "מה חדש?", "איך אתה?", "מה אתה עושה?", "אתה יכול לעזור לי?", "מה אתה יודע?"
]

sample_labels = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]  # תוויות לדוגמה: 1 = חיובי, 0 = שלילי

# יצירת Tokenizer כדי להפוך את הטקסטים למספרים
tokenizer = Tokenizer(num_words=10000)  # מוגבל ל-10,000 מילים השכיחות ביותר
tokenizer.fit_on_texts(sample_data)  # התאמנו את הטוקניזר לנתונים

# המרת הטקסטים לרשימות של מספרים
X = tokenizer.texts_to_sequences(sample_data)

# Padding להבטיח שכל טקסט יהיה באותו אורך
X = pad_sequences(X, padding='post')

# המרת התוויות למערך
y = np.array(sample_labels)

# פיצול הנתונים לסט אימון וסט בדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# יצירת מודל לדוגמה עם פרמטרים שונים
dl_model = DeepLearningModel(vocab_size=15000, embedding_dim=32, hidden_units=64)

# הדפסת סיכום המודל
dl_model.summary()

# דוגמת אימון - מאמנים את המודל עם הנתונים המעובדים
dl_model.train(X_train, y_train, epochs=10)

# הערכת המודל על סט הבדיקה
test_loss, test_accuracy = dl_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
