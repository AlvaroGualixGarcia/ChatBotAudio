import pandas as pd
import numpy as np
import torch
print(torch.__version__)
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import torch.nn.functional as F
import pygame

# Preguntas por categoría
preguntas_garantia = [
    "¿Cuál es la duración de la garantía para este producto?",
    "¿Qué cobertura ofrece la garantía?",
    "¿Puedo extender la garantía más allá del período estándar?",
    "¿Cuáles son los términos para reclamar la garantía?",
    "¿Hasta qué fecha puedo hacer uso de la garantía?",
    "¿Cómo se activa la garantía para nuevos productos?",
    "¿Qué procedimiento sigo si quiero hacer uso de la garantía?",
    "¿Cuánto tiempo después de la compra es válida la garantía?",
    "¿La garantía cubre daños accidentales?",
    "¿Hay alguna exclusión importante en la garantía?",
    "¿Qué documentos necesito para procesar un reclamo de garantía?",
    "¿Es transferible la garantía si vendo el producto?",
    "¿La garantía incluye soporte técnico?",
    "¿Cuántos meses de garantía ofrece la empresa por defectos de fábrica?",
    "¿Qué pasos debo seguir para validar la garantía?",
    "¿La garantía cubre componentes y accesorios?",
    "¿Se puede renovar la garantía al finalizar el periodo original?",
    "¿Qué tipo de daños están cubiertos por la garantía?",
    "¿Hay una garantía extendida disponible para compra?",
    "¿Quién es responsable de los gastos de envío en caso de reparación por garantía?",
    "¿Cuál es el tiempo estimado de respuesta para un reclamo de garantía?",
    "¿Cómo puedo verificar el estado de mi reclamo de garantía?",
    "¿Existe una política de devolución o reemplazo bajo la garantía?",
    "¿Puedo obtener un reembolso si el producto no puede ser reparado?",
    "¿La garantía aplica internacionalmente?",
    "¿Existen garantías especiales para ciertos componentes del producto?",
    "¿Qué sucede si el producto es reparado por un servicio no autorizado?",
    "¿Cómo afecta la modificación del producto a la garantía?",
    "¿La garantía cubre actualizaciones de software?",
    "¿Cuáles son las limitaciones de la cobertura de garantía?"
]

preguntas_servicio = [
    "¿Cuándo está disponible el servicio al cliente?",
    "¿Cómo puedo programar un servicio para mi producto?",
    "¿Existen servicios de emergencia disponibles?",
    "¿Cuál es el costo del servicio técnico?",
    "¿Dónde puedo encontrar una lista de los servicios ofrecidos?",
    "¿Qué horarios de atención tiene el servicio al cliente?",
    "¿El servicio incluye asistencia en el hogar?",
    "¿Cuánto tiempo tarda la prestación de un servicio típico?",
    "¿Qué opciones de servicio están disponibles para los clientes internacionales?",
    "¿Puedo cancelar o reprogramar un servicio programado?",
    "¿Qué políticas de privacidad aplican durante la prestación del servicio?",
    "¿Se ofrece algún tipo de garantía en los servicios proporcionados?",
    "¿Hay algún descuento disponible para servicios recurrentes?",
    "¿Cómo se manejan las quejas o problemas con un servicio realizado?",
    "¿Qué métodos de pago acepta el servicio?",
    "¿El servicio incluye consultas o diagnósticos gratuitos?",
    "¿Existen paquetes o promociones para múltiples servicios?",
    "¿Cómo puedo acceder al soporte técnico en línea?",
    "¿Hay un número directo para asistencia urgente?",
    "¿Se ofrece soporte multilingüe en el servicio al cliente?",
    "¿Cuáles son los procedimientos para resolver disputas de servicio?",
    "¿Se requiere algún tipo de preparación antes de recibir el servicio?",
    "¿Cómo puedo dar seguimiento a la calidad del servicio recibido?",
    "¿El servicio incluye alguna formación o instrucción para el usuario?",
    "¿Qué protocolos de seguridad se siguen durante la prestación del servicio?",
    "¿Existen restricciones de servicio basadas en la ubicación geográfica?",
    "¿Cómo puedo hacer una reserva de servicio en línea?",
    "¿Qué tan rápido es el tiempo de respuesta del servicio técnico?",
    "¿El servicio ofrece actualizaciones o mejoras para productos antiguos?",
    "¿Cómo puedo proporcionar retroalimentación sobre el servicio recibido?"
]

preguntas_experiencia = [
    "¿Cuántos años de experiencia tiene la empresa en esta industria?",
    "¿Qué experiencia previa tienen sus técnicos?",
    "¿Qué tan experimentado es el equipo de atención al cliente?",
    "¿Qué tipo de capacitación reciben sus empleados?",
    "¿La empresa ha recibido algún reconocimiento o premio en su campo?",
    "¿Qué tan satisfechos están los clientes anteriores con sus servicios?",
    "¿Puede proporcionar ejemplos de proyectos o servicios exitosos realizados?",
    "¿Cómo ha evolucionado la empresa a lo largo de los años?",
    "¿Qué políticas implementa la empresa para asegurar la calidad?",
    "¿Qué medidas toma la empresa para mantenerse actualizada en su sector?",
    "¿Cómo gestiona la empresa las situaciones de crisis o emergencia?",
    "¿Qué estrategias de mejora continua tiene implementadas la empresa?",
    "¿Cómo capacita la empresa a su personal para manejar tecnología avanzada?",
    "¿Qué tan bien está posicionada la empresa frente a sus competidores?",
    "¿Cómo maneja la empresa las quejas y sugerencias de los clientes?",
    "¿Qué procesos utiliza la empresa para asegurar la satisfacción del cliente?",
    "¿Cuál es la filosofía empresarial en cuanto a la atención al cliente?",
    "¿Cómo fomenta la empresa la innovación en sus servicios?",
    "¿Qué tanto enfoca la empresa en el desarrollo sustentable?",
    "¿Cuál ha sido la clave del éxito y la longevidad de la empresa en el mercado?",
    "¿Qué tipo de relaciones mantiene la empresa con sus proveedores y socios?",
    "¿Cómo asegura la empresa la confidencialidad y seguridad de la información del cliente?",
    "¿Cuáles son los principales desafíos que ha enfrentado la empresa y cómo los ha superado?",
    "¿Cuáles son los principales valores que guían la operación de la empresa?",
    "¿Cómo contribuye la empresa a la comunidad local?",
    "¿Qué políticas de diversidad e inclusión promueve la empresa?",
    "¿Cuáles son los objetivos a largo plazo de la empresa?",
    "¿Cómo maneja la empresa los cambios en las regulaciones del mercado?",
    "¿Qué tan adaptable es la empresa a las nuevas tendencias del mercado?",
    "¿Cuál es el impacto de la empresa en el sector en términos de innovación?"
]

preguntas_costo = [
    "¿Cuál es el costo total del producto incluyendo todos los servicios?",
    "¿Existen costos adicionales que debería considerar?",
    "¿Cómo se compara el costo de sus servicios con el de la competencia?",
    "¿Qué factores influyen en el precio final del servicio?",
    "¿Existen opciones de financiamiento o planes de pago disponibles?",
    "¿Qué descuentos o promociones están disponibles actualmente?",
    "¿Cómo se determina el precio de los servicios adicionales?",
    "¿Cuál es el rango de precios para los diferentes niveles de servicio ofrecidos?",
    "¿Hay algún costo por cancelación o modificación del servicio?",
    "¿Cuáles son las políticas de reembolso en caso de insatisfacción con el servicio?",
    "¿Existe algún costo oculto que debería saber?",
    "¿Cómo se calculan los impuestos aplicables al precio del servicio?",
    "¿Qué incluye exactamente el precio que estoy pagando?",
    "¿Existe algún beneficio o descuento por pago anticipado?",
    "¿Cuál es el costo por servicios de emergencia o fuera de horario?",
    "¿Existen tarifas reducidas para ciertos grupos o circunstancias?",
    "¿Cuál es la política de precios para clientes habituales?",
    "¿Cómo afectan las fluctuaciones del mercado al precio de sus servicios?",
    "¿Existen garantías de precio o ajustes de precio post-compra?",
    "¿Qué métodos de pago aceptan y cómo afectan al costo final?",
    "¿Cómo justifican el costo de sus servicios premium?",
    "¿Existen costos de mantenimiento o renovación que deba considerar?",
    "¿Cuál es el costo de la actualización o mejora de servicios existentes?",
    "¿Qué opciones de ahorro ofrece la empresa para presupuestos limitados?",
    "¿Cuál es el impacto de los costos de envío en el precio total?",
    "¿Cómo se determina el precio de los componentes o piezas de repuesto?",
    "¿Cuál es el costo por consultoría o asesoramiento especializado?",
    "¿Existen diferencias de precio según la ubicación o el mercado?",
    "¿Qué estrategias de precios utiliza la empresa para atraer nuevos clientes?",
    "¿Cómo se facturan los servicios adicionales o personalizados?"
]

# Concatenar todas las preguntas y etiquetas correspondientes
preguntas = preguntas_garantia + preguntas_servicio + preguntas_experiencia + preguntas_costo
categorias = (['garantia'] * len(preguntas_garantia) +
              ['servicio'] * len(preguntas_servicio) +
              ['experiencia'] * len(preguntas_experiencia) +
              ['costo'] * len(preguntas_costo))

# Crea DataFrame
df = pd.DataFrame({"Pregunta": preguntas, "Grupo": categorias})
etiqueta_grupos = {'garantia': 0, 'servicio': 1, 'experiencia': 2, 'costo': 3}
df['Grupo'] = df['Grupo'].map(etiqueta_grupos)

# Configuración del modelo y tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=4)

# Clase para el dataset y preparación de datos
class PreguntasDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

X_train, X_test, y_train, y_test = train_test_split(df['Pregunta'], df['Grupo'], test_size=0.2, random_state=42)
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)
train_dataset = PreguntasDataset(train_encodings, y_train.tolist())
test_dataset = PreguntasDataset(test_encodings, y_test.tolist())

# TrainingArguments y Trainer
training_args = TrainingArguments(
    output_dir='./results', num_train_epochs=10, per_device_train_batch_size=16,
    learning_rate=5e-5, warmup_steps=100, weight_decay=0.01, logging_dir='./logs',
    logging_steps=10, evaluation_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
results = trainer.evaluate()

# Respuestas por categoría y función de predicción
respuestas_categorias = {
    'garantia': "Le informamos que la garantía dura 12 meses.", 'servicio': "Nuestro servicio estará operativo en un plazo de 6 meses.",
    'experiencia': "Contamos con 25 años de experiencia en el mercado.", 'costo': "El servicio es de 30 euros al mes."
}
umbral_confianza = 0.95  # Ajusta este valor según lo que consideres adecuado

def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilidades = F.softmax(logits, dim=-1)
    max_prob, predictions = torch.max(probabilidades, dim=-1)
    if max_prob.item() < umbral_confianza:
        return "Por favor, ¿podría repetir la pregunta?"
    categoria_predicha = {v: k for k, v in etiqueta_grupos.items()}[predictions.item()]
    return respuestas_categorias[categoria_predicha]

# Funciones de audio
def hablar(mensaje):
    print(f"Chatbot: {mensaje}")
    tts = gTTS(text=mensaje, lang='es')
    filename = 'respuesta.mp3'
    tts.save(filename)
    # Inicia pygame
    pygame.mixer.init()
    # Carga y reproduce el archivo MP3
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    # Espera a que la música termine
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    os.remove(filename)
    
def activar_escucha():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Di 'siri' para activar...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listo para escuchar...")
        audio = recognizer.listen(source)
        try:
            texto = recognizer.recognize_google(audio, language='es-ES').lower()
            print("Escuchado:", texto)
            return texto
        except sr.UnknownValueError:
            print("No se pudo entender el audio")
            return ""
        except sr.RequestError:
            print("Error de servicio")
            return ""


# Interfaz de usuario
print("¡Bienvenido al chatbot de nuestra empresa!")
print("Puedes decir 'siri' en cualquier momento para hacer una consulta.")
while True:
    texto_activacion = activar_escucha()
    if "siri" in texto_activacion:
        print("¿En qué puedo ayudarte?")
        texto_pregunta = activar_escucha()
        if "salir" in texto_pregunta:
            hablar("¡Gracias por utilizar nuestro chatbot! Hasta luego.")
            break
        respuesta = predict(texto_pregunta)
        hablar(respuesta)
    elif "salir" in texto_activacion:
        hablar("¡Gracias por utilizar nuestro chatbot! Hasta luego.")
        break