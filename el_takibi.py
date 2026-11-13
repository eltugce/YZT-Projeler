import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ayarlar = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1,
)

el_bulucu = vision.HandLandmarker.create_from_options(ayarlar)

BAGLANTILAR = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

kamera = cv2.VideoCapture(0)

while True:
    basarili, kare = kamera.read()
    if not basarili:
        break

    kare = cv2.flip(kare, 1)


    rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    sonuc= el_bulucu.detect(mp_goruntu)

    if sonuc.hand_landmarks:
        el = sonuc.hand_landmarks[0]
        yukseklik, genislik, _ = kare.shape

        noktalar = []
        for n in el:
            x = int(n.x * genislik)
            y = int(n.y * yukseklik)
            noktalar.append((x,y))
        for a,b in BAGLANTILAR:
            cv2.line(kare,noktalar[a],noktalar[b],(255,255,255),2)

        for nokta in noktalar:
            cv2.circle(kare,nokta,5,(255,255,255),-1)
        acik = 0
        uclar = [8,12,16,20]
        eklemler = [6,10,14,18]

        for uc, eklem in zip(uclar, eklemler):
            if noktalar[uc][1] < noktalar[eklem][1]:
                acik += 1

        if noktalar[5][0] < noktalar[17][0]:
            if noktalar[4][0] < noktalar[3][0]:
                acik += 1

        else:
            if noktalar[4][0] > noktalar[3][0]:
                acik += 1

        cv2.putText(kare, f"Parmak: {acik}",(20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)

    cv2.imshow("El Takibi",kare)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
