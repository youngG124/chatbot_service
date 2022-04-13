# from flask import Flask, jsonify
# from flask_cors import CORS


# app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False
# CORS(app)

# # 모델 부분
# def makeAnswer(question) :
#     return 'd'

# @app.route('/')
# def default() :
#     return "답변 생성 서버입니다."

# # 질문에 대한 답변 return 하는 api
# @app.route('/<question>')
# def exportAnswer(question):
#     print(question)
#     return jsonify({"answer" : makeAnswer(question)})

# if __name__ == '__main__' :
#     app.run(debug=True, host="127.0.0.1", port=5000)

from jdi import predict

predict('약 복용에도 연령제한이 있나요')