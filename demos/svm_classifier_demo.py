from nlp.classfication.svm_classifier import SVMClassifier

if __name__ == '__main__':
    # svm_model = SVMClassifier('model/svm/model.pkl', 'data/imdb/aclImdb.txt', train=True)
    svm_model = SVMClassifier('model/svm/model.pkl')
    svm_model.predict(['i like it ! its very interesting', 'I don\'t like it, it\'s boring'])

    # 启动web
#     app = Sanic()
#
#     @app.route("/predict", methods=['POST', 'GET'])
#     async def predict(request):
#         """
#         采用restful接口的形式,获取分类结果
#         :param request: {
#                             "sentence": "待推测文本"
#                         }
#         :return:
#         """
#         nlp = request.json.get('sentence')
#         answer = svm_model.predict(nlp)
#
#         ans = answer[0]
#
#         return json({'category':ans})
#
#
# app.run(host="127.0.0.1", port=8000)
