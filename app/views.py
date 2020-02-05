from app import app
from flask import jsonify
from engine.testing_rnn import get_predictions
from datetime import datetime


@app.route('/')
@app.route('/index')
def index():
    txt = ["A WeWork shareholder has taken the company to court over the near-$1.7bn (£1.3bn) leaving package approved "
           "for ousted co-founder Adam Neumann."]
    score, predicated_labels, pred_list, labels = get_predictions(txt)
    data_dict = {label: format(pred_list[index], '.2f') for index, label in enumerate(labels)}
    return jsonify(data_dict)
