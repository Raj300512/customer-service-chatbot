from flask import Flask, render_template, request
from chatbot import chatbot_response          
from ai_chatbot import ai_chatbot_response    

app = Flask(__name__)

chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history

    if request.method == "POST":
        user_input = request.form["message"]

       
        bot_reply = chatbot_response(user_input)

        
        if bot_reply is None:
            bot_reply = ai_chatbot_response(
                "You are a professional customer support assistant. "
                "Answer clearly and politely.\nCustomer: " + user_input
            )

        chat_history.append(("user", user_input))
        chat_history.append(("bot", bot_reply))

    return render_template("index.html", chat_history=chat_history)

@app.route("/reset", methods=["POST"])
def reset():
    global chat_history
    chat_history.clear()
    return render_template("index.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
