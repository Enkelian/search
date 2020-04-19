import os.path
import string
import scipy.sparse as sparse
import numpy as np
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

max_depth = 2
res_dir = "res"
dir = "res" + os.sep + "stoner"
ps = PorterStemmer()


def get_terms(dir):
    terms = set({})
    document_count = 0
    documents_dict = {}
    i = 0
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            documents_dict[file] = i
            i += 1
            file_path = dir + os.sep + file
            document_count+=1
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
                words = text_to_words(text)
                for word in words:
                    terms.add(word)
    return np.sort(list(terms)), documents_dict


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def text_to_words(text):
    text = clear_text(text)
    words = word_tokenize(text)
    words = remove_stop_words(words)
    return words


def clear_text(text):
    text = "".join([c for c in text if c not in string.punctuation])
    return text.lower()


def remove_stop_words(words):
    stop_words = set(stopwords.words('english'))
    words = [ps.stem(w) for w in words if (w not in stop_words and is_ascii(w))]
    return words


terms, documents_dict = get_terms(dir)

terms_dict = {}
for i in range(len(terms)):
    terms_dict[terms[i]] = i

json.dump(terms_dict, open( res_dir + os.sep + "terms_dict.json", 'w' ))
json.dump(documents_dict, open( res_dir + os.sep + "documents_dict.json", 'w' ))


def get_term_by_document_matrix(terms_dict, dir, documents_dict):
    terms_count = len(terms_dict)
    documents_count = len(documents_dict)

    tbd = sparse.lil_matrix((terms_count, documents_count))
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            file_path = dir + os.sep + file
            with open(file_path, encoding="utf-8") as f:
                for word in text_to_words(f.read()):
                    tbd[terms_dict[word], documents_dict[file]] += 1
    return sparse.csc_matrix(tbd)


tbd = get_term_by_document_matrix(terms_dict, dir, documents_dict)
sparse.save_npz(res_dir + os.sep + 'tbd_no_idf.npz', tbd)


def apply_idf(tbd, terms_count, documents_count):
    idf = []
    tbd_csr = sparse.csr_matrix(tbd)
    i = 0
    for row in tbd_csr:
        doc_count = row.count_nonzero()
        idf_i = np.log(documents_count/doc_count)
        idf.append(idf_i)
        row = row * idf_i
    return idf, tbd_csr


idf, tbd_csr = apply_idf(tbd, len(terms_dict), len(documents_dict))


tbd_csc = sparse.csc_matrix(tbd_csr)
sparse.save_npz(res_dir + os.sep + 'tbd.npz', tbd_csc)


def get_col_norms(tbd, documents_count):
    col_norms = []

    for i in range(documents_count):
        col_norm = sparse.linalg.norm(tbd.getcol(i))
        col_norms.append(col_norm)
    return sparse.csc_matrix(col_norms)


col_norms = get_col_norms(tbd_csc, len(documents_dict))
sparse.save_npz(res_dir + os.sep + 'col_norms.npz', col_norms)


def normalize_tbd(tbd_csc):
    return normalize(tbd_csc, axis = 0)


tbd_csc = normalize_tbd(tbd_csc)
sparse.save_npz(res_dir + os.sep + 'tbd_normalized.npz', tbd_csc)


def apply_svd(tbd, k):
    svd = TruncatedSVD(n_components=k).fit(tbd)
    tbd = svd.transform(tbd)
    return tbd, svd


tbd = sparse.load_npz(res_dir + os.sep + 'tbd_normalized.npz')
tbd_trans, svd = apply_svd(tbd, 275)
sparse.save_npz(res_dir + os.sep + 'tbd_trans.npz', sparse.csc_matrix(tbd_trans))
sparse.save_npz(res_dir + os.sep + 'svd_compoments_.npz', sparse.csc_matrix(svd.components_))


def normalize_no_idf():
    tbd_no_idf = sparse.load_npz(res_dir + os.sep + 'tbd_no_idf.npz')
    tbd_no_idf = normalize_tbd(tbd_no_idf)
    sparse.save_npz(res_dir + os.sep + 'tbd_no_idf_normalized.npz', tbd_no_idf)


normalize_no_idf()

