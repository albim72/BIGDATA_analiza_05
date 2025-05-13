import pandas as pd
import hashlib

# Przykładowy DataFrame z danymi wrażliwymi
df = pd.DataFrame({
    'imie': ['Anna', 'Piotr', 'Jan'],
    'nazwisko': ['Kowalska', 'Nowak', 'Wiśniewski'],
    'pesel': ['44051401359', '80010112345', '91050609876'],
    'miasto': ['Warszawa', 'Kraków', 'Gdańsk'],
    'wiek': [34, 45, 29]
})

# Funkcja do maskowania (pseudonimizacji) danych – np. przez hash SHA-256
def pseudonimizuj_wartosc(wartosc):
    return hashlib.sha256(wartosc.encode()).hexdigest()

# Maskowanie kolumn imie, nazwisko i pesel
for kolumna in ['imie', 'nazwisko', 'pesel']:
    df[kolumna] = df[kolumna].apply(pseudonimizuj_wartosc)

print("Zanonimizowane dane:\n")
print(df)
