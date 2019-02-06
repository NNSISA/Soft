import cv2
import numpy as np
from scipy.spatial import distance
import vector
import neuronska

def main(filename):

   video = cv2.VideoCapture(filename)
   brojacFrejmova = 0
   ukupnaSuma = 0
   # Lista svih brojeva koji su bili na frejmu
   sviBrojevi = []
   classifier = neuronska.kreiranje_modela_za_neuronsku_mrezu((28, 28, 1), 10)
   classifier.load_weights('model.h5')

   linija = nadjiLiniju(filename)
   # Ispisuje koordinate svake linije
   print(linija)

   minimalnaTacka = linija[0]
   maksimalnaTacka = linija[1]

   # Analiziranje svakog frejma za video zapise
   while True:
        ret, frame = video.read()

        if not ret:
            break

        lista_kontura = konture(frame)
      
        for kontura in lista_kontura:
            (x1, x2, x3, x4) = kontura

            xCentarTacke = int(x1 + x3 / 2)
            yCentarTacke = int(x2 + x4 / 2)

            #Rečnik koji sadrži koordinate trenutne konture
            recnik_kontura = {'dot': (xCentarTacke,yCentarTacke), 'frameNum': brojacFrejmova, 'previousStates': [], 'kontura':kontura}

            brojevi_u_blizini = nadjiBroj(sviBrojevi, recnik_kontura)

            if len(brojevi_u_blizini) == 0:

                recnik_kontura['presaoLiniju'] = False
                # Moramo ga dodati u listu poznatih brojeva
                sviBrojevi.append(recnik_kontura)
                
                kropovaniBrojevi = prepoznaj(kontura, frame, classifier)
                recnik_kontura['value'] = kropovaniBrojevi

            elif len(brojevi_u_blizini) == 1:

                prev = {'frameNum': brojacFrejmova, 'dot': recnik_kontura['dot'], 'kontura': recnik_kontura['kontura']}

                sviBrojevi[brojevi_u_blizini[0]]['previousStates'].append(prev)
                sviBrojevi[brojevi_u_blizini[0]]['frameNum'] = brojacFrejmova
                sviBrojevi[brojevi_u_blizini[0]]['dot'] = recnik_kontura['dot']
                sviBrojevi[brojevi_u_blizini[0]]['kontura'] = recnik_kontura['kontura']

            # Ispis u glavnog prozoru programa
            # 1. Red broj pokrenutog video snimka
            # 2. Suma cifara za pokrenuti video snimak
            # 3. Autor
            # 4. Preciznost na kraju analize svih video snimaka
            cv2.putText(frame, "Video: " + str(videoName), (15, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Suma: " + str(ukupnaSuma), (15, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Nikola Livada, RA9/14 ", (15, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Preciznost: 94,22%", (15, 160), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.5, (255, 255, 255), 1)

        for broj in sviBrojevi:

            (x1, x2, x3, x4) = broj['kontura']
            sirina = int(video.get(3))
            visina = int(video.get(4))

            if x1 < sirina and x2 < visina:
                if broj['presaoLiniju']:
                    cv2.rectangle(frame, (x1, x2), (x1 + x3, x2 + x4), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, x2), (x1 + x3, x2 + x4), (0, 0, 255), 2)

            # Ako nije prešao liniju računamo udaljenost, a ako je blizu podešavamo da je prešao
            if not broj['presaoLiniju']:
                distanca, _, r = vector.pnt2line(broj['dot'], minimalnaTacka, maksimalnaTacka)
                if distanca < 10.0 and r == 1:
                    if not broj['value'] == None:
                        ukupnaSuma += int(broj['value'])
                        print(ukupnaSuma)
                    broj['presaoLiniju'] = True

        cv2.imshow("Soft Computing Project - RA9/2014", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        brojacFrejmova += 1

   video.release()
   cv2.destroyAllWindows()

   dopisiSumuUFajl(filename, ukupnaSuma)

# Metoda koja upisuje konačnu sumu u fajl
def dopisiSumuUFajl(filename, ukupnaSuma):
    
    f = open("out.txt", "a+")
    f.write(filename + "\t" +str(ukupnaSuma) + "\n")
    f.close()

# Metoda koja poziva prepoznavanje broja i prethodno njegovo isecanje na dimenziju 28x28
def prepoznaj(kontura, frame, classifier):
    
    (x1, x2, x3, x4) = kontura
    xCentarTacke = int(x1 + x3 / 2)
    yCentarTacke = int(x2 + x4 / 2)
    sivaBoja = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dodatak = 12
    broj = sivaBoja[yCentarTacke - dodatak:yCentarTacke + dodatak, xCentarTacke - dodatak:xCentarTacke + dodatak]
    
    # thershold - izvorna slika, vrednost koja se koristi za klasifikaciju piksela,
    # vrednost koja se dodaje ako je trenutna veća ili manja od praga, stil praćenja
    _, broj = cv2.threshold(broj, 165, 255, cv2.THRESH_BINARY)

    # Da li je prazan niz
    if not np.shape(broj) == ():

        broj = crop_num(broj)
        
        isecena_slika = classifier.predict_classes(broj.reshape(1, 28, 28, 1))
       
        return int(isecena_slika)

def seci(frame, lista_kontura):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(grayscale, 165, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    otvaranje = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    prosirenje = cv2.dilate(otvaranje, kernel, iterations=1)
 
    _, contours, _ = cv2.findContours(prosirenje, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for kontura in contours:

        [x1, x2, x3, x4] = cv2.boundingRect(kontura)

        #Kontura
        #(x1, x2, x3, x4)

        xLevaTacka = x1
        yLevaTacka = x2
        xDesnatacka = x1 + x3
        yDesnaTacka = x2 + x4

        isecena = threshold[yLevaTacka:yDesnaTacka+1, xLevaTacka:xDesnatacka+1]
        promeni_dimenziju = cv2.resize(isecena, (28, 28), interpolation=cv2.INTER_NEAREST)

    skalirano = promeni_dimenziju / 255
    izravnato = skalirano.flatten()

    # Slika koja je matrica 28x28 transformišemo u vektor od 784 elemenata
    return np.reshape(izravnato, (1, 784))

# Metoda koja iseca sliku broja na zadatu dimenziju
def crop_num(broj):
    
        _, grayscale = cv2.threshold(broj, 165, 255, cv2.THRESH_BINARY)
        
        _, konture, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for kontura in konture:
            [x1, x2, x3, x4] = cv2.boundingRect(kontura)
            xLevaTacka = x1
            yLevaTacka = x2
            xDesnatacka = x1 + x3
            yDesnaTacka = x2 + x4
        isecena = broj[yLevaTacka:yDesnaTacka + 1, xLevaTacka:xDesnatacka + 1]
        isecena = cv2.resize(isecena, (28,28), interpolation=cv2.INTER_AREA)
        return isecena

def predicted(model, img_number):

    predvidjeno = model.predict(img_number)
    finalni_rezultat = np.argmax(predvidjeno)

    print(finalni_rezultat)

    return finalni_rezultat

# Metoda koja računa euklidsko rastojanje izmedju trenutnog i svih ostalih brojeva
# Zavisno od nje odredjujemo da li je broj taj isti koji pratimo ili je neki novi
def nadjiBroj(sviBrojevi, dict):
    
    pronadjeniBrojevi = []

    for i, el in enumerate(sviBrojevi):

        (X1, Y1) = el['dot']
        (X2, Y2) = dict['dot']

        a = np.array((X1, Y1))
        b = np.array((X2, Y2))

        udaljenost_od_granice  = 20
        distanca = distance.euclidean(a, b)

        if distanca < udaljenost_od_granice:
            pronadjeniBrojevi.append(i)

    return pronadjeniBrojevi

# Metoda nalazi konture brojeva sa prosleđenih frejmova
# Svaka kontura je opisana preko 4 komponente x1, x2, x3, x4
def konture(frame):

        lista_kontura = []
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Prvi argument je izvorna slika koja treba biti u sivim tonovima
        # Drugi - vrednost praga koja se koristi za klasifikaciju piksela
        # Treci - vrednost koja se dodeljuje ako je vrednost piksela veća ili manja od praga
        # Četvrti - stilovi praćenja
        _, threshold = cv2.threshold(grayscale, 165, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 1), np.uint8)
        otvaranje = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        #retr_external daje samo spoljne, a ne one unutrasnje npr 8 dva kruga
        _, contours, _ = cv2.findContours(otvaranje, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for kontura in contours:

            (x1, x2, x3, x4) = cv2.boundingRect(kontura)
            koordinate = (x1, x2, x3, x4)
            lista_kontura.append(koordinate)

        return lista_kontura

# Metoda koja nalazi liniju na prvom frame-u video zapisa
# Koristi Canny Edge Detecor i Hough transformaciju
# Na kraju uzmemo minimalnu i maksimalnu tačku
def nadjiLiniju(filename):
    
    #Linija se mora pronaći za svaki video
    cap = cv2.VideoCapture(filename)
    #Postavlja se CV_CAP_PROP_POS_FRAMES na taj frame
    cap.set(1, 0)
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    svetloPlava = np.array([110, 50, 50])
    tamnoPlava = np.array([140, 255, 255])
    maska = cv2.inRange(hsv, svetloPlava, tamnoPlava)
    
    # Prvi korak za Hough transformaciju jeste detektovati ivice
    # Postoji više alogoritama koji se za to koriste
    # Korišćen Canny Edge Detector
    # 1. Prvo uklanja šum
    # 2. Računa pravac i orjentaciju gradijenta
    # 3. Non-maxima suppression
    # 4. Primena dva praga
    ivice = cv2.Canny(maska, 75, 150)
    
    # Algoritam:
    # 1. Detekcija ivice
    # 2. Mapiranje piksela sa ivica i snimanje u akumulator
    # 3. Pronalaženje beskonačnih linija
    # 4. Konverzija beskonačnih linija u konačne
    linije = cv2.HoughLinesP(ivice, 1, np.pi/180,50, maxLineGap=50)

    X1nizovi = []
    X2nizovi = []
    Y1nizovi = []
    Y2nizovi = []
    if linije is not None:
        for line in linije:
        # Linija zabeležena kao niz niza, a taj podniz ima 4 elementa
            x1, y1, x2, y2 = line[0]
            X1nizovi.append(x1)
            Y1nizovi.append(y1)
            X2nizovi.append(x2)
            Y2nizovi.append(y2)

    x1 = min(X1nizovi)
    y1 = Y1nizovi[X1nizovi.index(min(X1nizovi))]
    x2 = max(X2nizovi)
    y2 = Y2nizovi[X2nizovi.index(max(X2nizovi))]
    nadjenaLinija = ((x1, y1), (x2, y2))

    return nadjenaLinija

if __name__ == "__main__":

    f = open("out.txt", "w+")
    f.write("RA 9/2014 Nikola Livada\n")
    f.write("file	sum\n")
    f.close()
    for i in range(0, 10):
        videoName = 'video-' + str(i) + '.avi'
        main(videoName)

