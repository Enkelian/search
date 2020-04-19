from flask import Flask, render_template, request
from lab6.my_search import Search

app = Flask(__name__)
my_search = Search()


@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        query = request.form.get("qu")
        k = request.form.get("res")
        mode = request.form.get("mode")
        if k != '':
            k = int(k)
        else:
            k = 10
        results = my_search.find_documents(query, k, mode)
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
