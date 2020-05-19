"""
mnist_loader
~~~~~~~~~~~~
Moduł ładujący dane z bazy MNIST. Szczegółowe informacje na temat
formatu danych zawarte są w komentarzach w dalszej części modułu.
"""

import pickle
import gzip
import numpy as np


def load_data():
    """Funkcja zwracająca krotkę zawierającą trzy elementy: 'training data',
    'validation data' oraz 'test data'. Każdy z tych 3 elementów również jest krotką.
    'Training data' ma dwa elementy i opisuje 50 000 obrazków. Pierwszy element to numpy ndarray (w uproszczeniu macierz),
    która ma 50 000 wierszy. Zawarte są w niej dane z obrazków treningowych.
    Każdy wiersz reprezentuje jeden orazek i ma 784 elementy (28x28 pikseli z obrazków),
    jeden element w wierszu to jeden piksel w konkretnym przypadku treningowym.
    Macierz zawierające dane z obrazków ma więc rozmiar 50 000x784.
    Drugi element to również numpy ndarray (o rozmiarze 50 000x1), gdzie każdy element to
    spodziewany rezultat (po prostu cyfra) sieci neuronowej dla konkretnego przypadku testowego.
    'validation_data' i 'test_data' są podobne do 'training data', ale zawierają w sobie po
    10 000 obrazków. Format do jakiego dane są załadowane jest niemal idealny do wykorzystania
    przez nas w sieci neuronowej, tylko drobne zmiany są dokonywane w funkcji 'load_data_wrapper'.
    """

    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """Funkcja ta wykorzystując funkcję 'load_data' danye treningowe,
    walidacyjne oraz testowe w formacie odpowiednim do wykorzystania w sieci neuronowej.
    W przypadku 'training_data' ndarray, zawierający piksele z obrazka wejściowego,
    o wymiarach 50 000x784 zamieniany jest na listę
    zawierającą 50 000 ndarrayów, każdy o rozmiarze 784x1.
    Do każdego takiego ndarraya dołączany jest odpowiadający 10-elementowy wektor (funkcja zip)
    w którym jeden element jest równy 1.0 (oczekiwana cyfra jako rezultat), na pozostałych miejsach 0.0.
    Dla przykładu: jeśli oczekujemy, że dla danego obrazka sieć powinna odczytać cyfrę 3, to
    ten 10-elementowy wektor będzie wyglądał tak:
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    W przypadku 'validation_data' oraz 'test_data' do każdego ndarraya 784x1 dołączamy
    po prostu jedną cyfrę jako spodziewany rezultat.
    Finalnie, 'training_data', 'validation_data', 'test_data' to obiekty zip (w Pythonie
    2.x wynikiem funkcji zip była lista, w Pythonie 3.x iterator po krotkach, 'iteable object').
    """
    tr_d, va_d, te_d = load_data()
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    return training_data, validation_data, test_data


def vectorized_result(j):
    """Funkcja zwraca 10-elementowy wektor, gdzie element o indeksie j ma wartość 1, pozostałe 0.
    W ten sposób cyfra (0...9) konwertowana jest do właściwego formatu dla pożądanego wyjścia
    sieci neuronowej.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
