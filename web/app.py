import sys

from flask import Flask, request, jsonify
from flask.templating import render_template

from data.loader import Loader
from web.column_stat import Stat


if len(sys.argv) < 2:
    print("Usage: python main.py path-to-data-file")
    sys.exit(1)

app = Flask(__name__)
df = Loader.load(sys.argv[1])


def get_options(key, value=None):
    return [
        k.split(':')[1]
        for k in request.form
        if k.startswith(key + ':') and
        (value is None or request.form[k] == value)
    ]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', **{
        'columns': sorted([
            Stat(df, col) for col in df.columns
        ], key=lambda x: x.preference())
    })


@app.route('/generate', methods=['POST'])
def generate():
    params = {
        'exclude': get_options("exclude"),
        'onehot': get_options("onehot"),
        'dropna': get_options("dropna"),
        'copy': get_options("copy"),
        'datefeats': get_options("datefeats"),
        'minmax': get_options("scale", "minmax"),
        'robust': get_options("scale", "robust"),
    }

    return jsonify(**params)


app.run(debug=True)
