import cv2
import os

# Diretório para salvar os frames
caminho_frames = "frames/"
os.makedirs(caminho_frames, exist_ok=True)

def capturar_video():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    contador = 0
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rosto = frame[y:y+h, x:x+w]
            nome_arquivo = f"{caminho_frames}frame_{contador}.jpg"
            
            with open(nome_arquivo, 'wb') as arquivo:
                is_written, buffer = cv2.imencode('.jpg', rosto)
                if is_written:
                    arquivo.write(buffer)
            
            contador += 1
        
        cv2.imshow('Captura de Vídeo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

capturar_video()