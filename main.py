import cv2
from fer import FER

# Inicializa o FER (Detector de emoções)
emotion_detector = FER()

# Dicionário emoções
emotion_translation = {
    'angry': 'raiva',
    'disgust': 'nojo',
    'fear': 'medo',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'surpresa',
    'neutral': 'neutro'
}

# Abre a câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao capturar o vídeo.")
        break

    # Detectar emoções dentro da imagem
    emotions = emotion_detector.detect_emotions(frame)

    for emotion in emotions:
        # Pega as coordenadas do rosto
        x, y, w, h = emotion['box']
        
        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 70, 0), 2)  # Azul para rosto

        # Obter a emoção com maior pontuação
        dominant_emotion = emotion['emotions']
        emotion_label = max(dominant_emotion, key=dominant_emotion.get)

        # coleta do dicionario a emoção correspondente
        translated_emotion = emotion_translation.get(emotion_label, emotion_label)

        # Desenhar o esqueleto facial (pontos de referência)
        landmarks = [
            (x + w // 4, y + h // 4), 
            (x + w // 4 * 3, y + h // 4), 
            (x + w // 2, y + h // 2), 
            (x + w // 4, y + h // 4 * 3),
            (x + w // 4 * 3, y + h // 4 * 3)
        ]

        for point in landmarks:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Verde para os pontos do esqueleto

        # Exibir a emoção detectada 
        cv2.putText(frame, f'Emocao: {translated_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36,255,12), 2) 

    cv2.imshow('Deteccao de emocao', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()