import requests
import logging

# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def obtener_juegos_gratis():
    url = "https://store-site-backend-static.ak.epicgames.com/freeGamesPromotions?locale=es-ES"

    try:
        # Realiza la solicitud GET
        respuesta = requests.get(url)
        respuesta.raise_for_status()  # Verifica si hay errores HTTP
        datos = respuesta.json()

        # Extrae los juegos gratuitos
        juegos = datos["data"]["Catalog"]["searchStore"]["elements"]
        juegos_gratis = []
        for juego in juegos:
            precio_original = juego["price"]["totalPrice"]["originalPrice"] / 100
            precio_descuento = juego["price"]["totalPrice"]["discountPrice"] / 100
            # Verifica si el juego es gratuito
            if precio_descuento == 0:
                juegos_gratis.append({
                    "nombre": juego["title"],
                    "precio_original": precio_original,
                    "precio_descuento": precio_descuento,
                    "url_juego": f"https://store.epicgames.com/p/{juego['productSlug']}",
                    "imagen": juego["keyImages"][0]["url"] if juego.get("keyImages") else "Sin imagen"
                })

        return juegos_gratis if juegos_gratis else "No hay juegos gratuitos disponibles actualmente."

    except requests.exceptions.RequestException as e:
        logging.error(f"Error al conectarse a la API de Epic Games: {e}")
        return f"Error al conectarse a la API de Epic Games: {str(e)}"
    except KeyError as e:
        logging.error(f"Error al procesar los datos: clave no encontrada {e}")
        return f"Error al procesar los datos: clave no encontrada {str(e)}"
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        return f"Error inesperado: {str(e)}"


def obtener_proximos_juegos_gratis():
    url = "https://store-site-backend-static.ak.epicgames.com/freeGamesPromotions?locale=es-ES"

    try:
        # Realiza la solicitud GET
        respuesta = requests.get(url)
        respuesta.raise_for_status()  # Verifica si hay errores HTTP
        datos = respuesta.json()

        # Extrae los juegos y valida la estructura de datos
        juegos = datos.get("data", {}).get("Catalog", {}).get("searchStore", {}).get("elements", [])
        proximos_juegos = []

        for juego in juegos:
            promociones = juego.get("promotions", {})
            # Verifica si hay ofertas promocionales próximas
            if promociones and "upcomingPromotionalOffers" in promociones:
                ofertas = promociones["upcomingPromotionalOffers"]
                if ofertas and len(ofertas) > 0:
                    # Asegúrate de que las ofertas tengan datos válidos
                    oferta = ofertas[0].get("promotionalOffers", [])
                    if oferta and len(oferta) > 0:
                        proximos_juegos.append({
                            "nombre": juego.get("title", "Sin nombre"),
                            "inicio": oferta[0].get("startDate", "Sin fecha de inicio"),
                            "fin": oferta[0].get("endDate", "Sin fecha de fin"),
                            "url_juego": f"https://store.epicgames.com/p/{juego.get('productSlug', 'Sin URL')}",
                            "imagen": juego["keyImages"][0]["url"] if juego.get("keyImages") else "Sin imagen"
                        })

        return proximos_juegos if proximos_juegos else "No hay próximos juegos gratuitos disponibles actualmente."

    except requests.exceptions.RequestException as e:
        logging.error(f"Error al conectarse a la API de Epic Games: {e}")
        return f"Error al conectarse a la API de Epic Games: {str(e)}"
    except KeyError as e:
        logging.error(f"Error al procesar los datos: clave no encontrada {e}")
        return f"Error al procesar los datos: clave no encontrada {str(e)}"
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        return f"Error inesperado: {str(e)}"

if __name__ == "__main__":
    # Probar obtener juegos gratis
    juegos_gratis = obtener_juegos_gratis()
    print("Juegos gratis actuales:", juegos_gratis)

    # Probar obtener próximos juegos gratis
    proximos_juegos_gratis = obtener_proximos_juegos_gratis()
    print("Próximos juegos gratis:", proximos_juegos_gratis)
