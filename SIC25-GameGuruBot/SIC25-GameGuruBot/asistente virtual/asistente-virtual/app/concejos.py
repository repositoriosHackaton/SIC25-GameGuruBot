import json
import random
import os

def obtener_consejo():
    try:
        # Ruta al archivo consejos.json
        consejos_path = os.path.join(os.path.dirname(__file__), 'consejos.json')
        
        # Verifica si el archivo existe
        if not os.path.isfile(consejos_path):
            return "El archivo de consejos no existe. Por favor, verifica su ubicación."

        # Abre y carga el archivo JSON
        with open(consejos_path, "r", encoding="utf-8") as file:
            consejos = json.load(file)

        # Devuelve un consejo aleatorio o un mensaje si la lista está vacía
        return random.choice(consejos)["mensaje"] if consejos else "No hay consejos disponibles."

    except json.JSONDecodeError:
        return "Error al decodificar el archivo JSON. Verifica su formato."
    except Exception as e:
        return f"Error inesperado al cargar los consejos: {str(e)}"

# Prueba
if __name__ == "__main__":
    print("🔔 ¡Hora de una pausa! " + obtener_consejo())
