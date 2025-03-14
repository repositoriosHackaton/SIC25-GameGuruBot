import requests

def obtener_ofertas_gog():
    url = "https://www.gog.com/games/ajax/filtered?mediaType=game&price=discounted"

    try:
        # Realiza la solicitud GET a la API
        respuesta = requests.get(url)
        respuesta.raise_for_status()  # Verifica si hay errores HTTP
        datos = respuesta.json()

        # Asegúrate de que los productos estén presentes en los datos
        juegos = datos.get("products", [])
        if not juegos:
            return "No hay ofertas disponibles en este momento."

        # Procesa los primeros 5 juegos en oferta
        ofertas = []
        for juego in juegos[:5]:  # Limitar a los primeros 5 resultados
            nombre = juego.get("title", "Sin nombre")
            precio_original = juego.get("price", {}).get("baseAmount", "Desconocido")
            precio_descuento = juego.get("price", {}).get("amount", "Desconocido")
            descuento = juego.get("price", {}).get("discountPercentage", "0%")
            url_juego = f"https://www.gog.com{juego.get('url', '')}"
            imagen = f"https:{juego.get('image', '')}" if juego.get("image") else "Sin imagen"

            ofertas.append({
                "nombre": nombre,
                "precio_original": precio_original,
                "precio_descuento": precio_descuento,
                "descuento": descuento,
                "url_juego": url_juego,
                "imagen": imagen
            })

        return ofertas

    except requests.exceptions.RequestException as e:
        return f"Error al conectarse a la API de GOG: {str(e)}"
    except ValueError as e:
        return f"Error al procesar la respuesta de GOG: {str(e)}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"
