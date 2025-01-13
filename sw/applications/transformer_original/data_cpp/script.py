import re

def suma_valores_entre_corchetes(linea):
    # Utilizamos expresiones regulares para encontrar los valores entre corchetes
    valores = re.findall(r'\[(\d+)\]', linea)
    
    # Sumamos los valores encontrados
    suma = sum(map(int, valores))
    
    return suma

def procesar_archivo(archivo):
    try:
        with open(archivo, 'r') as file:
            total_suma = 0
            for linea in file:
                suma_linea = suma_valores_entre_corchetes(linea)
                total_suma += suma_linea

        print(f"La suma total de los valores entre corchetes en {archivo} es: {total_suma}")

    except FileNotFoundError:
        print(f"El archivo {archivo} no se encontró.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Reemplaza 'output.txt' con el nombre de tu archivo
procesar_archivo('data.cpp')
