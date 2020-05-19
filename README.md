## Zadania dla uczestników demonstracji o uczeniu maszynowym w Pythonie

### Wymagania
- Python w wersji 3.5+ (rekomendowany 3.7) :snake:
- Biblioteka numpy


```sh
pip3 install numpy
```
Jeśli masz problemy z zainstalowaniem numpy możesz poszukać rozwiązania tutaj: <https://www.scipy.org/install.html>
***
### Zadania
1. W katalogu 'src' znajduje się 5 plików. W tym zadaniu interesuje Cię tylko jeden - 'teach_network.py'. Umożliwia on stworzenie sieci neuronowej i zaobserwowanie procesu jej uczenia się. Aby stworzyć sieć neuronową o wybranej przez Ciebie liczbie neuronów użyj:
    ```sh
    net = network.Network([784, 30, 10])
    ```
    Pierwszy element listy to liczba neuronów w warstwie wejściowej (musi być ich 784!), ostatni to liczba neuronów w warstwie wyjściowej (musi być ich 10!). Wszystko co wpiszesz "w środku" będzie określało liczbę warstw ukrytych i neuronów w nich, w powyższym przypadku jest to jedna warstwa ukryta z 30 neuronami. Spróbuj poeksperymentować z liczbą warstw ukrytych oraz ilością neuronów w tych warstwach. Aby sprawdzić jak sieć radzi sobie z wprowadzonymi zmianami po prostu uruchom 'teach_network.py' nie zmieniając nic więcej w skrypcie. Zaobserwuj jak liczba warstw i neuronów wpływa na efektywność sieci. 
    ```sh
    net.SGD(training_data=training_data, epochs=5, mini_batch_size=10, eta=3.0, test_data=test_data)
    ```
    Po sprawdzeniu jak liczba neuronów wpływa na skuteczność sieci spróbuj modyfikować inne parametry sieci:
    * rozmiar pojedynczej 'porcji danych' - trzeci argument w metodzie SGD (10 w powyższym przykładzie)
    * współczynnik uczenia się - czwarty argument w metodzie SGD (3.0)
    
    Zaobserwuj jak sieć zachowuje się przy zmianach tych parametrów.
    Drugi argument to liczba epok, przez które sieć będzie się uczyła. Jeśli chcesz możesz go zmienić, ale 5 epok jest optymalnym rozwiązaniem, zdążysz zauważyć generalną efektywność sieci nie tracąc przy tym zbyt dużo czasu.   
    ***Przy zmienianiu parametrów nie musisz robić tego bardzo skrupulatnie - sprawdź kiedy dany parametr jest zdecydowanie zbyt duży lub zbyt mały "psując" sieć (poprawność sieci zdecydowanie poniżej 90%). Zapisz swoje wnioski (bardzo ogólne, powinieneś się zmieścić w mniej niż 10 zdaniach) na temat tego jak zmiana liczby neuronów, parametru uczenia się i rozmiaru pojedynczej 'porcji danych' wpływa na działanie sieci.***  
    Jeśli chcesz możesz poszukać optymalnych parametrów sieci, dla których osiągnie ona skuteczność ~95% (dla tej dosyć prostej sieci osiągnięcie wyższych wyników jest raczej niemożliwe). Do osiągnięcia takiego wyniku będziesz prawdopodobnie potrzebował  więcej niż 5 epok. 
    > ***Uwaga***: Wagi i progi w sieci są inicjalizowane losowo - z tego powodu dla tych samych parametrów sieci może się okazać, że dla różnych podejść uzyskasz zupełnie inne wyniki (w pierwszych epokach może być to różnica nawet kilku punktów procentowych).

2. W pliku 'network.py' znajduje się pusta metoda klasy Network 'network_result'. Twoim zadaniem jest uzupełnić tę metodę tak, aby dla konkretnych danych testowych zwracała ona listę zawierającą tylko rezultaty sieci neuronowej. Możesz posiłkować się metodą 'evaluate', która jest bardzo podobna do pożądanej metody 'network_result'.
Jeśli chcesz sprawdzić czy metoda zwraca to, co chcemy możesz dopisać do pliku 'teach_network.py' wywołanie metody 'network_result' np. w taki sposób:
    ```sh
    print(net.network_result(test_data))
    ```
    > ***Protip 1***: Metoda 'load_data_wrapper' z 'mnist_loader.py' zwraca dane jako obiekty zip (w uproszczeniu jest to iterator przechodzący po kolejnych krotkach, w naszym przypadku po krotkach o długości 2 zawierających dane wejściowe oraz pożądany rezultat dla konkretnych przypadków testowych.). Oznacza to tyle, że w instrukcji 'for * in *' nie możemy użyć tylko danych wejściowych 'x', lecz krotkę (x, y) mimo tego, że pożądany rezultat sieci 'y' jest w tej metodzie niepotrzebny. W przeciwnym przypadku program nie zadziała.
    
    > ***Protip 2***: Nie musisz za każdym uruchomieniem skryptu tracić czas, żeby tworzyć i uczyć nową sieć. W katalogu 'data' zawarte są zapisane parametry dla sieci już nauczonej (skuteczność ~95%). Wykomentuj więc linie odpowiadające za tworzenie oraz uczenie się sieci (net = network... oraz net.SGD). Zamiast tego dodaj następującą linię:
    
    ```sh
    net = network.load('../data/network_params')
    ```
    
    > ***Uwaga***: Jako że test_data jest obiektem zip (czyli iteratorem po kolejnych krotkach) to po "skorzystaniu" z tych danych nie będzie można wykorzystać ich ponownie (iterator nie będzie wskazywał na pierwszą krotkę). Jeśli będziesz uczył sieć używając test_data do monitorowania skuteczności, a następnie będziesz chciał sprawdzić rezultat metody 'network_result' używając również test_data wynikiem będzie pusta lista, nawet jeśli napisałeś metodę poprawnie. Możesz poradzić sobie z tym problemem na trzy sposoby: wczytać wszystkie dane jeszcze raz, tak jak na początku skryptu, nie podawać w ogóle argumentu 'test_data' w celu monitorowania przy wywoływaniu metody 'net.SGD' lub w ogóle nie uczyć sieci o czym mowa w drugim protipie.
3. Pora przejść do ulepszonej wersji naszej sieci znajdującej się w pliku 'network2.py'. Jednym z problemów istniejących w pierwszej sieci było nieoptymalne inicjalizowanie wag. Nie wchodząc w szczegóły, spowalniało to uczenie się sieci. Metoda, która implementuje tę inicjalizację to 'large_weight_initializer'. Wagi tam są losowane z rozkładu gaussowskiego z wartością średnią 0 oraz odchyleniem standardowym 1. Aby wyeliminować problem tzw. 'learning slowdown' wystarczy, aby odchylenie standardowe wynosiło nie 1, a 1/√n, gdzie n to liczba wag wchodzących do konkretnego neuronu. Jako, że w naszej sieci pomiędzy kolejnymi warstwami każdy neuron z danej warstwy połączony jest z każdym neuronem z poprzedniej warstwy dokładnie raz, to liczba wag wchodzących do danego neuronu to po prostu liczba neuronów w poprzedniej warstwie. Twoim zadaniem jest zmodyfikowanie metody 'default_weight_initializer', która na razie niczym nie różni się od 'large_weight_initializer', tak aby inicjalizowała ona wagi z odchyleniem standardowym 1/√n oraz średnią 0. Aby sprawdzić czy Twoje rozwiązanie działa uruchom skrypt 'teach_improved.py'. 
    ```sh
    net.large_weight_initializer()
    ```
    Powyższa linia wymusza inicjalizowanie wag z odchyleniem 1. Jeśli ją wykomentujesz wagi będą inicjalizowane za pomocą metody 'default_weight_initializer', czyli tej, którą powinieneś zmodyfikować. Jeśli poprawnie wykonasz zadanie, sieć która wykorzystała optymalniejsze inicjalizowanie wag powinna się szyciej uczyć (np. w 2 epoce osiągać taką skuteczność, jak sieć wykorzystująca nieoptymalne inicjalizowanie w 5 epoce). Pamiętaj jednak, że inicjalizowanie wag wciąż jest losowe, więc możesz uzyskać inne rezultaty dla różnych uruchomień skryptu.
    > ***Protip***: Całe zadanie polega tak naprawdę na dopisaniu odpowiedniego dzielenia w jednym miejscu. Może przydać Ci się również funkcja z biblioteki NumPy zaimportowanej w skrypcie jako np - np.sqrt().
    
    > ***Uwaga***: Sieć utworzona w pliku 'teach_improved.py' ma 20 neuronów w warswie ukrytej. Nie jest to optymalna liczba neuronów (najlepsza skuteczność tej sieci osiągana jest dla około 100 neuronów), jednak znacznie przyspiesza proces uczenia i wciąż pozwala na zaobserwowanie różnic w uczeniu się sieci w zależności od sposobu inicjalizacji. Używamy też tylko 100 przypadków testowych z 'validation_data', a nie wszystkich 10 000, po to aby szybciej móc porównać rezultaty sieci. Reszta parametrów jest ustalona optymalnie, ale możesz spróbować je zmieniać jeśli chcesz.
4. Ostatnim zadaniem jest zaimplementowanie funkcjonalności wczesnego zatrzymywania sieci. Polega ona na tym, że sieć przestanie się uczyć, niezależnie od zdefiniowanej początkowo liczby epok przez które miała się uczyć, jeśli wykryje brak poprawy skuteczności sieci w kolejnych n epokach (gdzie n to opcjonalny parametr podawany przy wywoływaniu metody net.SGD). Część tej funkcjonalności jest już zaimplementowana, Twoim zadaniem jest ją dokończyć. Zawarta jest ona w metodzie SGD, po komentarzach w kodzie zorientujesz się gdzie powinieneś dopisać swój kod. Żeby sprawdzić czy wczesne zatrzymywanie działa wystarczy, że uruchomisz skrypt "teach_improved.py" podając argument 'early_stopping_n' np. tak:
    ```sh
    net.SGD(training_data = training_data, epochs=50, mini_batch_size=10, eta=0.1, lmbda=5.0, 
        evaluation_data=list(validation_data)[:100], monitor_evaluation_accuracy=True, early_stopping_n=5)
    ```
    Powyższa linia spowoduje, że sieć przestanie się uczyć jeśli przez 5 epok nie nastąpi poprawa skuteczności (o ile dobrze zaimplementowałeś funkcjonalność wczesnego zatrzymywania).  