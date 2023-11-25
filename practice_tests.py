import math


class CountVectorizer:
    """
    Класс для преобразования текстов в матрицу частот слов.

    Методы
    ------
    fit_transform(corpus: List[str]) -> List[List[int]]:
        Преобразует корпус текстов в матрицу частот слов.

    get_feature_names() -> List[str]:
        Возвращает список уникальных слов.
    """

    def __init__(self):
        self.feature_names = []

    def fit_transform(self, corpus):
        word_count = {}
        for text in corpus:
            words = text.lower().split()
            for word in words:
                if word not in word_count:
                    word_count[word] = 0

        self.feature_names = sorted(word_count)

        matrix = []
        for text in corpus:
            words = text.lower().split()
            row = [words.count(word) for word in self.feature_names]
            matrix.append(row)

        return matrix

    def get_feature_names(self):
        return self.feature_names


def tf_transform(count_matrix):
    """
    Преобразует матрицу частот слов в матрицу TF (частота).
    """
    tf_matrix = []
    for row in count_matrix:
        word_count = sum(row)
        tf_matrix.append([round(freq / word_count, 3)
                         if word_count else 0 for freq in row])
    return tf_matrix


def idf_transform(count_matrix):
    """
    Преобразует матрицу частот слов в вектор IDF (обратная частота).
    """
    num_docs = len(count_matrix)
    idf = []
    for i in range(len(count_matrix[0])):
        doc_with_term = sum(1 for doc in count_matrix if doc[i] > 0)
        idf.append(round(math.log((num_docs + 1)
                                  / (doc_with_term + 1)) + 1, 3))
    return idf


class TfidfTransformer:
    """
    Класс для преобразования матрицы частот слов в матрицу TF-IDF.
    """

    def fit_transform(self, count_matrix):
        tf_matrix = tf_transform(count_matrix)
        idf_vector = idf_transform(count_matrix)
        tfidf_matrix = []
        for row in tf_matrix:
            tfidf_row = [round(tf * idf, 3)
                         for tf, idf in zip(row, idf_vector)]
            tfidf_matrix.append(tfidf_row)
        return tfidf_matrix


class TfidfVectorizer(CountVectorizer):
    """
    Класс для преобразования корпуса текстов в матрицу TF-IDF.
    """

    def fit_transform(self, corpus):
        count_matrix = super().fit_transform(corpus)
        return TfidfTransformer().fit_transform(count_matrix)


# Пример использования
if __name__ == "__main__":
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(tfidf_matrix)
