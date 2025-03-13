import json
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from .utils import preprocess, read_db, write_db, detect_language, translate_text
from .config import Config
import openai
from wikipediaapi import Wikipedia

# Configurar la API de OpenAI
openai.api_key = ""
# Configuración para la subida de archivos
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configuración para la base de datos
DB_PATH = os.path.join(os.path.dirname(__file__), 'db.json')

# Funciones para leer y escribir db.json
def read_db():
    with open(DB_PATH, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_db(data):
    with open(DB_PATH, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Cargar el modelo y el tokenizer
model_folder = 'model_folder'

# Cargar datos de entrenamiento desde un archivo JSON
try:
    with open('training_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    data = []

# Asegúrate de que 'data' sea una lista
data = list(data)

def save_training_data():
    with open('training_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)

def get_gpt4o_mini_answer(query, msg):
    print(f"Buscando en GPT-4o-mini: {query}")
    client = openai.OpenAI(api_key=openai.api_key)
    
    system_message = {
        "role": "system",
        "content": "Eres GameGuru Bot, un asistente virtual creado por el equipo Mugiwaras para ayudar con información sobre videojuegos electrónicos."
    }
    
    if not msg:
        msg.append(system_message)
        welcome_message = {
            "role": "assistant",
            "content": "¡Hola! Soy GameGuru Bot, tu asistente virtual creado por el equipo Mugiwaras. Estoy aquí para ayudarte con cualquier consulta relacionada con videojuegos electrónicos. ¿En qué puedo ayudarte hoy?"
        }
        msg.append(welcome_message)
    
    msg.append({"role": "user", "content": query})
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=msg,
        max_completion_tokens=4096
    )
    answer = completion.choices[0].message.content.strip()
    
    non_satisfactory_answers = [
        "Como soy una inteligencia artificial, no tengo la capacidad",
        "No tengo información",
        "No tengo acceso",
        "No tengo una respuesta para eso",
        "Lo siento, pero no tengo información",
        "Lo siento, pero no tengo información sobre eso",
        "No tengo información sobre eso",
        "No tengo información sobre ese tema",
        "No tengo información sobre este tema",
        "No tengo información sobre ese juego",
        "No tengo información sobre este juego",
        "No tengo información sobre ese evento",
        "No tengo información sobre este evento",
        "No tengo información sobre ese jugador",
        "No tengo información sobre este jugador",
        "No tengo información sobre esa estrategia",
        "No tengo información sobre esa pregunta",
        "No tengo información sobre esa consulta",
        "No tengo información sobre esa operación",
        "No tengo información sobre esa fecha",
        "No tengo información sobre esa operación matemática",
        "No tengo información sobre esa operación aritmética"
    ]
    
    if any(phrase in answer for phrase in non_satisfactory_answers):
        print("No tengo una respuesta para eso. ¿Cuál debería ser la respuesta?")
        new_answer = input("Guardar respuesta: ")
        data.append((query, new_answer))
        questions, answers = zip(*data)
        preprocessed_questions = [preprocess(q) for q in questions]
        model.fit(preprocessed_questions, answers)
        save_training_data()
        print("Nueva pregunta y respuesta añadidas y modelo reentrenado.")
        msg.append({"role": "assistant", "content": new_answer})
        return new_answer
    elif answer:
        print("Respuesta encontrada en GPT-4o-mini")
        return answer
    else:
        print("Respuesta no encontrada en GPT-4o-mini")
        return "Lo siento, no tengo información sobre eso."

def load_model():
    if os.path.exists(model_folder) and os.listdir(model_folder):  # Si la carpeta existe y no está vacía
        try:
            model = DistilBertForSequenceClassification.from_pretrained(model_folder)
            tokenizer = DistilBertTokenizer.from_pretrained(model_folder)
            print("Modelo cargado desde el almacenamiento.")
        except Exception as e:
            print(f"Error al cargar el modelo guardado: {e}")
            print("Inicializando modelo desde cero.")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    else:
        print("Carpeta 'model_folder' no encontrada. Inicializando modelo desde cero.")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        os.makedirs(model_folder, exist_ok=True)
        model.save_pretrained(model_folder)
        tokenizer.save_pretrained(model_folder)
        print("Modelo guardado por primera vez en 'model_folder'.")
    return model, tokenizer

model, tokenizer = load_model()

def train_model():
    db_data = read_db()
    questions_db = db_data.get('questions', {})
    questions = [q_data['content'] for q_data in questions_db.values()]
    answers = [q_data['answer'] for q_data in questions_db.values() if q_data['answer'] is not None]

    # Mapeo de respuestas únicas a índices numéricos
    unique_answers = list(set(answers))
    answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}
    labels = [answer_to_index[answer] for answer in answers]

    # Configuración del modelo con etiquetas correctas
    num_labels = len(unique_answers)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )

    # Tokenización y configuración del dataset
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=64)  # Reduce max_length
    labels = torch.tensor(labels)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

    # Dividir el dataset en entrenamiento y prueba
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduce batch_size
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Configuración del optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Valor recomendado para DistilBERT

    # Entrenamiento
    for epoch in range(3):  # Reduce el número de épocas
        total_loss = 0
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Época {epoch+1}/3 completada. Pérdida promedio: {total_loss:.4f}")

    # Guardar el modelo
    try:
        model.save_pretrained(model_folder, safe_serialization=False)  # Forzar el guardado en formato binario
        tokenizer.save_pretrained(model_folder)
        print("Modelo guardado correctamente en 'model_folder'.")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

def evaluate_validation_loss(model, validation_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(validation_loader)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

def calculate_metrics(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    return accuracy, f1, recall

def save_model(model):
    model.save_pretrained(model_folder)
    tokenizer.save_pretrained(model_folder)

def get_response(question, user_id):
    # Detectar idioma y traducir si es necesario
    lang = detect_language(question)
    if lang != 'en':
        question = translate_text(question, src_lang=lang, dest_lang='en')

    # Preprocesar la pregunta
    preprocessed_question = preprocess(question)

    # Mover modelo y tokenizer al dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(preprocessed_question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Evaluar la pregunta usando el modelo local
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()

    print(f"Pregunta procesada: {preprocessed_question}")
    print(f"Predicción: {predicted_class}, Probabilidades: {probabilities}")

    # Buscar en la base de datos
    db_data = read_db()
    questions_db = db_data.get('questions', {})
    for q_id, q_data in questions_db.items():
        if q_data['content'].lower() == question.lower() and q_data['user_id'] == user_id:
            response = q_data['answer']
            break
    else:
        response = "Lo siento, no tengo información sobre eso."

    # Traducir la respuesta al idioma original
    if lang != 'en':
        question = translate_text(question, src_lang=lang, dest_lang='en')
        response = translate_text(response, src_lang='en', dest_lang=lang)
        
    return response

def get_wikipedia_summary(query):
    print(f"Buscando en Wikipedia: {query}")
    wiki_wiki = Wikipedia(language='es', user_agent='asistente-virtual/1.0')
    page = wiki_wiki.page(query)
    if page.exists():
        print("Página encontrada en Wikipedia")
        return page.summary[:500]  # Devuelve un resumen de 500 caracteres
    else:
        print("Página no encontrada en Wikipedia")
        return None

def get_response_with_fallback(question, user_id):
    # Paso 1: Intenta obtener una respuesta de GPT-4
    try:
        gpt4_response = get_gpt4o_mini_answer(question, [])
        if gpt4_response:
            return gpt4_response
    except Exception as e:
        print(f"Error al consultar GPT-4: {e}")

    # Paso 2: Si GPT-4 falla, busca en Wikipedia
    try:
        wikipedia_summary = get_wikipedia_summary(question)
        if wikipedia_summary:
            return wikipedia_summary
    except Exception as e:
        print(f"Error al buscar en Wikipedia: {e}")

    # Paso 3: Si Wikipedia falla, usa el modelo local
    try:
        local_response = get_response(question, user_id)
        if local_response:
            return local_response
    except Exception as e:
        print(f"Error al consultar el modelo local: {e}")

    # Paso 4: Si todo falla, registra la pregunta en la base de datos
    try:
        db_data = read_db()
        questions_db = db_data.get('questions', {})
        new_question_id = len(questions_db) + 1
        questions_db[new_question_id] = {
            "content": question,
            "answer": None,  # Sin respuesta por ahora
            "user_id": user_id
        }
        write_db(db_data)
        return "Lo siento, no tengo una respuesta ahora. He registrado tu consulta para analizarla más tarde."
    except Exception as e:
        print(f"Error al guardar la pregunta: {e}")
        return "Lo siento, no tengo una respuesta y tampoco pude registrar tu consulta."

def get_game_strategy(game_name):
    return f"Estrategia para {game_name}"

def get_performance_analysis(player_id):
    return f"Análisis de desempeño para el jugador {player_id}"

def get_esports_event_info(event_name):
    return f"Información del evento {event_name}"

def retrain_model():
    db_data = read_db()
    questions_db = db_data.get('questions', {})
    questions = [q_data['content'] for q_data in questions_db.values()]
    answers = [q_data['answer'] for q_data in questions_db.values()]
    
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = torch.tensor([answers.index(label) for label in answers])
    
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        val_loss = evaluate_validation_loss(model, test_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            model.save_pretrained(model_folder)
            tokenizer.save_pretrained(model_folder)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                break