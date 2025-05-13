from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64

# Dane do zaszyfrowania
dane = "To są dane medyczne pacjenta: Jan Kowalski, 45 lat."

# Wygeneruj losowy 16-bajtowy klucz AES (dla AES-128)
klucz = get_random_bytes(16)
print("Klucz (hex):", klucz.hex())

# Inicjalizuj szyfrowanie w trybie CBC
cipher = AES.new(klucz, AES.MODE_CBC)
iv = cipher.iv  # wektor inicjalizujący

# Zakoduj i zaszyfruj dane
dane_zaszyfrowane = cipher.encrypt(pad(dane.encode(), AES.block_size))
dane_zaszyfrowane_b64 = base64.b64encode(iv + dane_zaszyfrowane)

print("\nZaszyfrowane dane (base64):", dane_zaszyfrowane_b64.decode())

# Deszyfrowanie
dane_zaszyfrowane_full = base64.b64decode(dane_zaszyfrowane_b64)
iv_odczytane = dane_zaszyfrowane_full[:AES.block_size]
cipher_dec = AES.new(klucz, AES.MODE_CBC, iv=iv_odczytane)
dane_odszyfrowane = unpad(cipher_dec.decrypt(dane_zaszyfrowane_full[AES.block_size:]), AES.block_size)

print("\nOdszyfrowane dane:", dane_odszyfrowane.decode())
