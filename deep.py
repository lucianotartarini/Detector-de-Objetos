import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
net  = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                                "MobileNetSSD_deploy.caffemodel")
clases = {0:"Fondo",
          1:"Avion",
          2:"Bicicleta",
          3:"Ave",
          4:"Bote",
          5:"Botella",
          6:"Colectivo",
          7:"Auto",
          8:"Gato",
          9:"Silla",
          10:"Vaca",
          11:"Mesa",
          12:"Perro",
          13:"Caballo",
          14:"Moto",
          15:"Persona",
          16:"Planta en una maceta",
          17:"Oveja",
          18:"Sillon",
          19:"Tren",
          20:"Pantalla"}

cv2.namedWindow("deteccion", cv2.WINDOW_NORMAL)

while True:
    res, frame = cam.read()
    fRedimensionado = cv2.resize(frame, (300,300))
    blob = cv2.dnn.blobFromImage(fRedimensionado, 0.007843, (300,300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detecciones = net.forward()
    h, w = fRedimensionado.shape[:2]
    
    for i in range(detecciones.shape[2]):
        confidence = detecciones[0,0,i,2]
        
        if confidence > 0.75:
            idClase = int(detecciones[0,0,i,1])
            xSup = int(detecciones[0,0,i,3] * h)
            ySup = int(detecciones[0,0,i,4] * w)
            xInf = int(detecciones[0,0,i,5] * h)
            yInf = int(detecciones[0,0,i,6] * w)
            
            hScaleFactor = frame.shape[0]/300.0
            wScaleFactor = frame.shape[1]/300.0
            
            xInf = int(wScaleFactor * xInf)
            yInf = int(hScaleFactor * yInf)
            xSup = int(wScaleFactor * xSup)
            ySup = int(hScaleFactor * ySup)
            
            cv2.rectangle(frame, (xSup, ySup), (xInf, yInf), (0,255,0), 2)
            etiqueta = clases[idClase] + ":" + str(confidence)
            cv2.putText(frame, etiqueta, (xSup, ySup), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,255), 2)
    cv2.imshow("deteccion", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


























