// Función para mostrar el mensaje de bienvenida
function showWelcomeMessage() {
    const welcomeMessage = "¡Hola! Soy GameGuru Bot, tu asistente virtual creado por el equipo Mugiwaras. Estoy aquí para ayudarte con cualquier consulta relacionada con videojuegos competitivos o esports. ¿En qué puedo ayudarte hoy?";
    addMessage(welcomeMessage, 'assistant');
}

// Llamar a la función de bienvenida al cargar la página
window.addEventListener('DOMContentLoaded', (event) => {
    showWelcomeMessage();
});

document.getElementById('send-button').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value;
    const userId = '12345'; // Reemplaza esto con el ID del usuario real si tienes uno
    if (userInput.trim() === '') return;

    // Mostrar mensaje del usuario inmediatamente
    addMessage(userInput, 'user');

    // Respuesta del asistente después de un pequeño retraso
    getAssistantResponse(userInput, userId).then(assistantResponse => {
        console.log('Respuesta del asistente:', assistantResponse); // Registro de depuración
        // Mostrar la respuesta del asistente después de un retraso
        setTimeout(() => {
            if (assistantResponse.includes("Lo siento, no tengo información sobre eso.")) {
                // Mostrar mensaje emergente para guardar la respuesta solo si no se encuentra la respuesta
                let newAnswer = prompt("No tengo una respuesta para eso. ¿Cuál debería ser la respuesta?");
                if (newAnswer) {
                    saveNewAnswer(userInput, newAnswer, userId);
                    addMessage(newAnswer, 'assistant');
                } else {
                    addMessage(assistantResponse, 'assistant');
                }
            } else {
                addMessage(assistantResponse, 'assistant');
            }
        }, 100); // Reducido el tiempo de retraso a 100 ms
    });

    // Limpiar el campo de entrada
    document.getElementById('user-input').value = '';
});

// Función para guardar la nueva respuesta en el servidor
async function saveNewAnswer(query, answer, userId) {
    try {
        await fetch('/save_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: query, response: answer, user_id: userId })
        });
    } catch (error) {
        console.error('Error al guardar la nueva respuesta:', error);
    }
}

// Función para obtener la respuesta del asistente
async function getAssistantResponse(input, userId) {
    try {
        const response = await fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: input, user_id: userId })
        });
        const data = await response.json();
        console.log('Datos recibidos del servidor:', data); // Registro de depuración
        return data.response;
    } catch (error) {
        console.error('Error al obtener la respuesta del asistente:', error);
        return "Lo siento, ocurrió un error al obtener la respuesta.";
    }
}

// Evento para abrir el selector de archivos
document.getElementById('upload-file-button').addEventListener('click', function() {
    document.getElementById('file-input').click();
});

// Evento para abrir el selector de audios
document.getElementById('upload-audio-button').addEventListener('click', function() {
    document.getElementById('audio-input').click();
});

// Manejar la subida de archivos
document.getElementById('file-input').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file);
    }
});

// Manejar la subida de audios
document.getElementById('audio-input').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file);
    }
});

// Función para subir archivos al servidor
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (file.type.startsWith('image/')) {
            addImage(result.filepath, 'assistant');
        } else {
            addMessage(result.message, 'assistant');
        }
    } catch (error) {
        console.error('Error al subir el archivo:', error);
        addMessage('Hubo un error al subir el archivo.', 'assistant');
    }
}

// Función para mostrar la imagen en el chat
function addImage(filepath, sender) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    const imgElement = document.createElement('img');
    imgElement.src = filepath;
    imgElement.alt = 'Uploaded Image';
    imgElement.style.maxWidth = '100%';
    messageElement.appendChild(imgElement);
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Desplazar hacia abajo
}

// Función para mostrar el mensaje letra por letra
function addMessage(message, sender) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    messageElement.style.opacity = 0; // Comienza invisible
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Desplazar hacia abajo
    console.log('Mensaje agregado al chat:', message); // Registro de depuración

    // Efecto de escritura letra por letra solo para el asistente
    if (sender === 'assistant') {
        typeWriter(message, messageElement, 40); // 40 ms de retraso entre letras
    } else {
        messageElement.innerHTML = message; // Mostrar el mensaje completo para el usuario
        messageElement.style.opacity = 1; // Hacer visible el texto al final
    }
}

// Función para el efecto de escritura
function typeWriter(text, element, delay) {
    let index = 0;

    function type() {
        if (index < text.length) {
            element.innerHTML += text.charAt(index);
            index++;
            setTimeout(type, delay);
        } else {
            element.style.opacity = 1; // Hacer visible el texto al final
        }
    }

    type();
}

// Agregar evento para enviar mensaje con la tecla Enter
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Evitar el comportamiento por defecto de Enter
        document.getElementById('send-button').click(); // Simular clic en el botón de enviar
    }
});
