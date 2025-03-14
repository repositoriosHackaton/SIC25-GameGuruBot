import os
import json
import nltk
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import re
import logging
import threading
import wikipediaapi
import openai
import emoji  # Importar la biblioteca emoji

# Configuración de logging
logging.basicConfig(level=logging.INFO)

# Intentar importar `Config` y sobreescribir configuraciones predeterminadas
try:
    from app.config import Config
    ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS
    LANGUAGE = Config.LANGUAGE
    IMPORTANT_WORDS = Config.IMPORTANT_WORDS
except ImportError as e:
    logging.warning(f"No se pudo importar `Config`. Usando valores predeterminados: {e}")
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'avi', 'mp4', 'mkv'}
    LANGUAGE = 'spanish'
    IMPORTANT_WORDS = {"cómo", "qué", "por qué", "cuál", "cuándo", "dónde", "quién"}

# Verificar y descargar datos de NLTK si es necesario
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt')
if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
    nltk.download('stopwords')

# Stopwords personalizadas basadas en el idioma configurado
try:
    stop_words = set(stopwords.words(LANGUAGE))
    logging.info(f"Stopwords cargadas para el idioma '{LANGUAGE}'")
except Exception as e:
    logging.error(f"Error al cargar stopwords: {e}")
    stop_words = set()  # Fallback

# Manejo de archivos
def allowed_file(filename):
    """
    Verifica si un archivo tiene una extensión permitida.
    """
    if '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Operaciones con la base de datos
db_lock = threading.Lock()

def read_db():
    """
    Lee el archivo de base de datos JSON. Si no existe, lo crea.
    """
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.json')
    try:
        if not os.path.exists(db_path):
            initial_data = {"questions": {}}
            with open(db_path, 'w', encoding='utf-8') as file:
                json.dump(initial_data, file, ensure_ascii=False, indent=4)
        with open(db_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        logging.error(f"Error al decodificar JSON en la base de datos: {e}")
        return {"questions": {}}  # Fallback: estructura vacía
    except Exception as e:
        logging.error(f"Error general al leer la base de datos: {e}")
        return {"questions": {}}

def write_db(data):
    """
    Escribe datos en el archivo de base de datos JSON.
    """
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.json')
    try:
        with db_lock, open(db_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except json.JSONDecodeError as e:
        logging.error(f"Error al codificar datos JSON para la base de datos: {e}")
    except Exception as e:
        logging.error(f"Error general al escribir en la base de datos: {e}")

# Preprocesamiento de texto
def preprocess(text):
    """
    Preprocesa un texto: convierte a minúsculas, elimina caracteres especiales, emojis y espacios extra.
    """
    text = text.lower()
    text = re.sub(r'[^a-zñáéíóúü0-9\s]', '', text)  # Permitir caracteres en español
    text = emoji.replace_emoji(text, replace='')  # Eliminar emojis
    text = re.sub(r'\s+', ' ', text).strip()  # Espacios adicionales
    return text

# Detección de idioma
def detect_language(text):
    """
    Detecta el idioma de un texto.
    """
    try:
        return detect(text)
    except LangDetectException as e:
        logging.warning(f"No se pudo detectar el idioma del texto: {e}")
        return "unknown"
    except Exception as e:
        logging.error(f"Error inesperado al detectar el idioma: {e}")
        return "unknown"

# Traducción de texto
def translate_text(text, src_lang=None, dest_lang=LANGUAGE):
    """
    Traduce un texto de un idioma fuente a un idioma destino usando GoogleTranslator.
    """
    try:
        if src_lang not in GoogleTranslator().get_supported_languages(as_dict=True).values():
            logging.warning(f"Idioma no soportado: {src_lang}. Usando inglés ('en').")
            src_lang = 'en'
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except Exception as e:
        logging.error(f"Error al traducir el texto: {e}")
        return text

# Función para obtener resumen de Wikipedia
def get_wikipedia_summary(query):
    """
    Busca en Wikipedia y devuelve un resumen del contenido (500 caracteres).
    """
    print(f"Buscando en Wikipedia: {query}")
    wiki = wikipediaapi.Wikipedia(language=LANGUAGE, user_agent="asistente_virtual/1.0")
    page = wiki.page(query)
    if page.exists():
        print("Página encontrada en Wikipedia")
        return page.summary[:500]  # Devuelve los primeros 500 caracteres
    else:
        print("Página no encontrada en Wikipedia")
        return None

# Función para consultas a OpenAI
def call_openai_api(question):
    """
    Consulta la API de OpenAI para generar respuestas contextuales.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Cambia a un modelo compatible según tu suscripción
            messages=[
                {"role": "system", "content": "Eres un asistente virtual listo para ayudar con preguntas contextuales."},
                {"role": "user", "content": question}
            ],
            max_tokens=4096,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error al consultar la API de OpenAI: {e}")
        return "Lo siento, ocurrió un error al obtener la respuesta."