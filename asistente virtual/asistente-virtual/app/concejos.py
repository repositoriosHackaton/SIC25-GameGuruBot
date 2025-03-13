import json
import random
import os

def obtener_consejo():
    try:
        # Ruta al archivo consejos.json
        consejos_path = os.path.join(os.path.dirname(__file__), 'consejos.json')
        with open(consejos_path, "r", encoding="utf-8") as file:
            consejos = json.load(file)
            return random.choice(consejos)["mensaje"] if consejos else "No hay consejos disponibles."
    except Exception as e:
        return f"Error al cargar los consejos: {str(e)}"

# Prueba
if __name__ == "__main__":
    print("🔔 ¡Hora de una pausa! " + obtener_consejo())