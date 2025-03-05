from ai_core.self_learning import self_learning
from models.machine_learning import ml_model
from models.deep_learning import dl_model

class ChatBot:
    def get_response(self, user_input):
        print(f"📥 קלט מהמשתמש: {user_input}")  # בדיקה

        response_ml = None
        response_dl = None

        # ניסיון לקבל תשובה ממודל למידת מכונה
        try:
            response_ml = ml_model.predict(user_input)
            print(f"🤖 תשובת ML: {response_ml}")
        except Exception as e:
            print(f"⚠️ שגיאה ב-ML: {e}")

        # ניסיון לקבל תשובה ממודל למידה עמוקה
        try:
            response_dl = dl_model.predict(user_input)
            print(f"🤖 תשובת DL: {response_dl}")
        except Exception as e:
            print(f"⚠️ שגיאה ב-DL: {e}")

        # בחירת תשובה מתוך ML ו-DL
        final_response = response_ml if response_ml else response_dl

        # אם יש תשובה, נלמד ממנה
        if final_response:
            try:
                self_learning.learn_from_user(user_input, final_response)
                print(f"🧠 לימוד מהמשתמש: קלט - {user_input}, תשובה - {final_response}")
            except Exception as e:
                print(f"⚠️ שגיאה בלמידה: {e}")
        
        # אם אין תשובה, מציעים תשובה ברירת מחדל
        return final_response or "לא הבנתי אותך, נסה שוב."

# יצירת מופע של הצ'אט בוט
chatbot = ChatBot()
