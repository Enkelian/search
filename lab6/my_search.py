import os.path
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
import json
from .text_preprocessor import TextPreprocessor
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')


class Search:
    def __init__(self):
        self.res_dir = "res"
        self.dir = self.res_dir + os.sep + "stoner"
        self.terms_dict = json.load(open(self.res_dir + os.sep + "terms_dict.json"))
        self.documents_dict = json.load(open(self.res_dir + os.sep + "documents_dict.json"))
        self.text_preprocessor = TextPreprocessor()
        self.tbd = sparse.load_npz(self.res_dir + os.sep + 'tbd.npz')
        self.tbd_normalized = sparse.load_npz(self.res_dir + os.sep + 'tbd_normalized.npz')
        self.tbd_trans = sparse.load_npz(self.res_dir + os.sep + 'tbd_trans.npz')
        self.col_norms = sparse.load_npz(self.res_dir + os.sep + 'col_norms.npz')
        self.svd_components = sparse.load_npz(self.res_dir + os.sep + 'svd_compoments_.npz')

    def query_to_bow(self, query):

        query_words = self.text_preprocessor.text_to_words(query)
        query_words = self.clean_query(query_words)

        if not query_words:
            return []

        query_bow = sparse.lil_matrix((len(self.terms_dict), 1))

        for word in query_words:
            query_bow[self.terms_dict[word], 0] += 1
        return query_bow

    def clean_query(self, query_words):
        query_words = [word for word in query_words if word in self.terms_dict]
        return query_words

    def find_documents(self, query, k, mode):
        query_bow = self.query_to_bow(query)
        if query_bow == []:
            return []

        query_bow_t = sparse.csr_matrix(query_bow.transpose())

        results = []

        if mode == "Not normalized":
            query_norm = sparse.linalg.norm(query_bow)
            tbd = self.tbd
        elif mode == "Normalized":
            query_bow_t = normalize(query_bow_t, axis=0)
            tbd = self.tbd_normalized
        else:
            query_bow_t = normalize(query_bow_t, axis=1)
            tbd = self.tbd_trans
            similarities = query_bow_t.dot(tbd)
            similarities = similarities.dot(self.svd_components).getrow(0).toarray()[0]

        for document in self.documents_dict.keys():
            i = self.documents_dict[document]

            if mode == "Not normalized":
                col = tbd.getcol(i)
                res = query_bow_t.dot(col)[0, 0] / (self.col_norms.getcol(i)[0, 0] * query_norm)
            elif mode == "Normalized":
                col = tbd.getcol(i)
                res = query_bow_t.dot(col)[0, 0]
            else:
                res = similarities[i]

            if i < k:
                results.append((res, document))
            else:
                results = sorted(results)
                if res > results[0][0]:
                    results[0] = (res, document)
        return self.get_title_text_link(results)

    def get_title_text_link(self, results):
        title_text_link = []
        for res in results:
            file_path = self.dir + os.sep + res[1]
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
                title = res[1].replace('.txt', '')
                title_text_link.append((title, text, wiki_wiki.page(title).fullurl))
        return title_text_link
