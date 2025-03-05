import os
from flask import Flask, render_template, request, jsonify
from ai_core.chat_bot import chatbot

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        print(f"📥 Received request data: {data}")  # בדיקה אם הנתונים מתקבלים

        if not data or "message" not in data:
            print("❌ Error: Missing 'message' in request")
            return jsonify({"error": "Missing message"}), 400

        user_input = data["message"].strip()
        if not user_input:
            print("❌ Error: Empty message received")
            return jsonify({"error": "Empty message"}), 400

        response = chatbot.get_response(user_input)
        print(f"👤 User: {user_input} | 🤖 Response: {response}")  # בדיקה מה מוחזר

        return jsonify({"response": response})

    except Exception as e:
        print(f"🔥 Server Error: {e}")  # הדפסת שגיאה למסוף השרת
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # שימוש במשתנה סביבה PORT (Render דורש את זה)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
