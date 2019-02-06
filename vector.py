import math

# Kompjuterska grafika obično se bavi linijama u 3D prostorima 
# kao što su definisane tačke koje obezbjeđuju koordinate
# početaka i krajeva linija. 
# Najkraća razdaljina između ovih tačaka i segmenata može biti dužina
# okoline koja je povezana sa tačkom i linijom ili može biti udaljnost
# od početne ili kraće linije.

# Tačka
def dot(v, w):
    x1, x2 = v
    x3, x4 = w
    return x1 * x3 + x2 * x4

# Dužina  
def length(v):
    x1, x2 = v
    return math.sqrt(x1 * x1 + x2 * x2)

# Vektor  
def vector(b, e):
    x1, x2 = b
    x3, x4 = e
    return (x3 - x1, x4 - x2)

# Jedinični vektor  
def unit(v):
    x1, x2 = v
    mag = length(v)
    return (x1 / mag, x2 / mag)

# Vraća udaljenost  
def distance(p0, p1):
    return length(vector(p0,p1))

# Skaliranje  
def scale(v, sc):
    x1, x2 = v
    return (x1 * sc, x2 * sc)
  
def add(v, w):
    x1, x2 = v
    x3, x4 = w
    return (x1 + x3, x2 + x4)

# Data je linija sa tačkama 'start' i'end' i koordinate tačke
# 'pnt' koja vraća najmanju udaljenost od pnt do line i koordinate
# najbliže tačke od linije.
#
# 1  Pretvaranje segmenta linije u vektor ('line_vec').
# 2  Kreiranje vektora i spavanje sa pnt ('pnt_vec').
# 3  Pronalaženje dužine vektora linije ('line_len').
# 4  Konvertovanje line_vec u jedinični vektor ('line_unitvec').
# 5  Skaliranje pnt_vec pomoću line_len ('pnt_vec_scaled').
# 6  Dobijanje dot-a od line_unitvec i pnt_vec_scaled ('t').
# 7  Pretvaranje tu raspon od 0 do 1.
# 8  Koristi se t da bismo dobili najbližu lokaciju od line do end
#    vektora pnt_vec_scaled ('nearest').
# 9  Računanje udaljenosti od najbližeg do pnt_vec_scaled.
# 10 Prevođenje najbliže nazad start/end line. 

  
def pnt2line(pnt, start, end):
    
    # Pretvaranje linije u vektor
    # Koordinate vektora koji predstavljaju tačku su u odnosu na početak linije
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    # Prilagođavanje oba vektora dižini linije
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    # Izračunavanje tačkastog proizvoda skaliranih vektora
    # Vrednost odgovara rastojanju prikazanog crnom bojom
    # duž vektora jedinice prikazane zelenom bojom
    t = dot(line_unitvec, pnt_vec_scaled)

    # T se nalazi u opsegu od 0 do 1
    # Treba prilagoditi vektor sa T da bi pronašli najbližu lokaciju sa zelenom bojom
    # Računa se udaljenost od najbliže lokacije do kraja vektora tačke    
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)

def pnt2line2(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)
