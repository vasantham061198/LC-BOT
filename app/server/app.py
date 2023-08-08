from flask import Flask, request

from server.LC_Bot import question_answer

app=Flask(__name__)


@app.route("/ask", methods=["POST"])
def ask():
    question = {"question": request.form["ask"]}
    result = question_answer(question)
    return {"answer":result["answer"]}


def generate_app():
    return app
