import requests

def obtener_juegos_gratis():
    url = "https://store-site-backend-static.ak.epicgames.com/freeGamesPromotions?locale=es-ES"
    respuesta = requests.get(url)
    datos = respuesta.json()
    
    juegos = datos["data"]["Catalog"]["searchStore"]["elements"]
    juegos_gratis = []
    for juego in juegos:
        nombre = juego["title"]
        precio_original = juego["price"]["totalPrice"]["originalPrice"] / 100
        precio_descuento = juego["price"]["totalPrice"]["discountPrice"] / 100
        url_juego = f"https://store.epicgames.com/p/{juego['productSlug']}"
        imagen = juego["keyImages"][0]["url"] if juego["keyImages"] else "Sin imagen"
        
        juegos_gratis.append({
            "nombre": nombre,
            "precio_original": precio_original,
            "precio_descuento": precio_descuento,
            "url_juego": url_juego,
            "imagen": imagen
        })
    
    return juegos_gratis

def obtener_proximos_juegos_gratis():
    url = "https://store-site-backend-static.ak.epicgames.com/freeGamesPromotions?locale=es-ES"
    respuesta = requests.get(url)
    datos = respuesta.json()

    juegos = datos["data"]["Catalog"]["searchStore"]["elements"]
    proximos_juegos_gratis = []
    for juego in juegos:
        if juego["promotions"]["upcomingPromotionalOffers"]:
            nombre = juego["title"]
            inicio = juego["promotions"]["upcomingPromotionalOffers"][0]["startDate"]
            proximos_juegos_gratis.append({
                "nombre": nombre,
                "inicio": inicio
            })
    
    return proximos_juegos_gratis