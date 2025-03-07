import logging
import concurrent.futures
import time
from ai_core.self_learning import self_learning
from models.machine_learning import ml_model
from models.deep_learning import dl_model
from typing import Optional

# הגדרת קונפיגורציה ללוגים
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        self.response_ml = None
        self.response_dl = None

    def _get_ml_response(self, user_input: str) -> Optional[str]:
        """
        מבצע חיזוי באמצעות מודל למידת מכונה.
        """
        start_time = time.time()
        try:
            response = ml_model.predict(user_input)
            logger.info(f"🤖 ML תשובה: {response}, זמן ריצה: {time.time() - start_time:.2f} שניות")
            return response
        except Exception as e:
            logger.error(f"⚠️ שגיאה ב-ML: {e.__class__.__name__}: {e}")
            return None

    def _get_dl_response(self, user_input: str) -> Optional[str]:
        """
        מבצע חיזוי באמצעות מודל למידת עומק.
        """
        start_time = time.time()
        try:
            response = dl_model.predict(user_input)
            logger.info(f"🤖 DL תשובה: {response}, זמן ריצה: {time.time() - start_time:.2f} שניות")
            return response
        except Exception as e:
            logger.error(f"⚠️ שגיאה ב-DL: {e.__class__.__name__}: {e}")
            return None

    def _learn_from_user(self, user_input: str, final_response: str) -> None:
        """
        לומד מהמשתמש את הקלט והתשובה הסופית.
        """
        try:
            self_learning.learn_from_user(user_input, final_response)
            logger.info(f"🧠 למידה מהמשתמש: קלט - {user_input}, תשובה - {final_response}")
        except Exception as e:
            logger.error(f"⚠️ שגיאה בלמידה עצמית: {e.__class__.__name__}: {e}")

    def get_response(self, user_input: str) -> str:
        """
        מקבל קלט מהמשתמש, מריץ חיזוי ממודלים שונים ובוחר תשובה.
        """
        logger.info(f"📥 קלט מהמשתמש: {user_input}")

        # הרצת שני המודלים במקביל עם max_workers=2
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_ml = executor.submit(self._get_ml_response, user_input)
            future_dl = executor.submit(self._get_dl_response, user_input)

            response_ml = future_ml.result()
            response_dl = future_dl.result()

        # בחירת התשובה הסופית: אם יש תשובה מ-ML ניקח אותה, אחרת נשתמש ב-DL
        final_response = response_ml or response_dl or "אנחנו עובדים על זה! נסה שוב מאוחר יותר."

        # למידה עצמית אם התקבלה תשובה תקינה
        if final_response not in ["אנחנו עובדים על זה! נסה שוב מאוחר יותר.", None]:
            self._learn_from_user(user_input, final_response)

        return final_response

# יצירת מופע של הצ'אט בוט
chatbot = ChatBot()

# דוגמה לקריאה לפונקציה
response = chatbot.get_response("שלום")
print(response)
