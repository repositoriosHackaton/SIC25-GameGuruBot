from flask import Flask, jsonify
from app.routes import app_routes
from apscheduler.schedulers.background import BackgroundScheduler
from app.models import retrain_model, load_model
import os
from dotenv import load_dotenv
import torch
import openai
from app.config import Config
from app.concejos import obtener_consejo
from app.epicgames import obtener_juegos_gratis, obtener_proximos_juegos_gratis
from app.gog import obtener_ofertas_gog

# Configurar la API de OpenAI
openai.api_key = 'pon tu clave aquí'

# Cargar variables de entorno
if not os.path.exists('.env'):
    print("Advertencia: No se encontró el archivo .env.")
else:
    load_dotenv()

# Ruta absoluta al archivo .env
env_path = os.path.join(os.path.dirname(__file__), '.env')

# Intentar cargar el archivo .env o crearlo si no existe
if not os.path.exists(env_path):
    print(f"Advertencia: No se encontró el archivo .env en la ruta {env_path}. Creándolo con valores predeterminados...")
    with open(env_path, 'w') as f:
        f.write('FLASK_ENV=development\n')
        f.write('UPLOAD_FOLDER=uploads\n')
        f.write('ALLOWED_EXTENSIONS=txt,pdf,png,jpg,jpeg,gif,mp3,wav,avi,mp4,mkv\n')
        f.write('OPENAI_API_KEY=tu_clave_aqui\n')
        f.write('LEARNING_RATE=0.00005\n')
        f.write('BATCH_SIZE=16\n')
        f.write('NUM_EPOCHS=25\n')
        f.write('PATIENCE=3\n')
        f.write('LANGUAGE=spanish\n')
        f.write('SECRET_KEY=mi_clave_secreta\n')
    print(f"Archivo .env creado en {env_path}. Por favor revisa y edita los valores según sea necesario.")

# Cargar el archivo .env
load_dotenv(env_path)
print(f"Archivo .env cargado desde {env_path}.")

# Crear la aplicación Flask
def create_app():
    app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
    app.config.from_object('app.config.Config')
    app.register_blueprint(app_routes)

    # Configurar la carpeta de subida de archivos
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

    # Cargar modelo y tokenizer como parte del contexto de la aplicación
    try:
        with app.app_context():
            app.model, app.tokenizer = load_model()
            app.logger.info("Modelo y tokenizer cargados correctamente.")
    except Exception as e:
        app.logger.error(f"Error al cargar el modelo o tokenizer: {e}")
        raise RuntimeError(f"Error crítico al cargar el modelo: {e}")

    return app

# Agregar nuevas rutas
def register_routes(app):
    @app.route('/juegos_epic', methods=['GET'])
    def juegos_epic():
        try:
            # Obtener juegos gratuitos actuales
            juegos_gratis = obtener_juegos_gratis()
            # Obtener próximos juegos gratuitos
            proximos_juegos = obtener_proximos_juegos_gratis()

            respuesta = {
                "juegos_gratis": juegos_gratis,
                "proximos_juegos_gratis": proximos_juegos
            }
            return jsonify(respuesta), 200
        except Exception as e:
            app.logger.error(f"Error en la ruta /juegos_epic: {e}")
            return jsonify({'error': 'Ocurrió un error al obtener la información de juegos.'}), 500

if __name__ == '__main__':
    app = create_app()

    # Registrar rutas adicionales
    register_routes(app)

    # Verificar disponibilidad de CUDA/cuDNN
    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("Dispositivo CUDA:", torch.cuda.get_device_name(0))
            print("Cantidad de GPUs disponibles:", torch.cuda.device_count())
        except Exception as e:
            print(f"Error al acceder a información de CUDA: {e}")
    else:
        print("Advertencia: CUDA no está disponible. El modelo se ejecutará en CPU.")
    print("cuDNN habilitado:", torch.backends.cudnn.enabled)

    # Programador de tareas para el retraining del modelo
    scheduler = BackgroundScheduler()
    try:
        scheduler.add_job(retrain_model, 'interval', days=1)
        scheduler.start()
        print("Scheduler de retraining iniciado correctamente.")
    except Exception as e:
        print(f"Error al iniciar el scheduler: {e}")

    # Registrar cierre limpio del scheduler
    import atexit
    atexit.register(lambda: scheduler.shutdown())

    # Iniciar la aplicación Flask
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error al iniciar la aplicación Flask: {e}")

    # Prueba de obtener consejo
    print("🔔 ¡Hora de una pausa! " + obtener_consejo())

    # Prueba de obtener juegos gratis
    print("🎮 Juegos gratis en Epic Games:")
    juegos_gratis = obtener_juegos_gratis()
    for juego in juegos_gratis:
        print(f"{juego['nombre']} - Antes: {juego['precio_original']}€, Ahora: {juego['precio_descuento']}€")
        print(f" {juego['url_juego']}")
        print(f" {juego['imagen']}\n")

    # Prueba de obtener próximos juegos gratis
    print("🎮 Próximos juegos gratis en Epic Games:")
    proximos_juegos_gratis = obtener_proximos_juegos_gratis()
    for juego in proximos_juegos_gratis:
        print(f"{juego['nombre']} será gratis desde: {juego['inicio']}")

    # Prueba de obtener ofertas de GOG
    print("🎮 Ofertas en GOG:")
    ofertas_gog = obtener_ofertas_gog()
    for oferta in ofertas_gog:
        print(f"{oferta['nombre']} - Antes: {oferta['precio_original']}€, Ahora: {oferta['precio_descuento']}€ (-{oferta['descuento']}%)")
        print(f"🔗 {oferta['url_juego']}")
        print(f"🖼️ {oferta['imagen']}\n")