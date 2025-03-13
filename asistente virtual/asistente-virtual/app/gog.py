import requests

def obtener_ofertas_gog():
    url = "https://www.gog.com/games/ajax/filtered?mediaType=game&price=discounted"
    respuesta = requests.get(url)
    datos = respuesta.json()
    
    juegos = datos["products"]
    ofertas = []
    for juego in juegos[:5]:  # Muestra los 5 primeros juegos en oferta
        nombre = juego["title"]
        precio_original = juego["price"]["baseAmount"]
        precio_descuento = juego["price"]["amount"]
        descuento = juego["price"]["discountPercentage"]
        url_juego = f"https://www.gog.com{juego['url']}"
        imagen = f"https:{juego['image']}"
        
        ofertas.append({
            "nombre": nombre,
            "precio_original": precio_original,
            "precio_descuento": precio_descuento,
            "descuento": descuento,
            "url_juego": url_juego,
            "imagen": imagen
        })
    
    return ofertas