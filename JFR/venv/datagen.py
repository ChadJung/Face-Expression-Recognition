import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

i = 0
while cap.isOpened():

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        i += 1
        # print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]  # roi: region of interest
        img_space = 75
        roi_color = frame[y-img_space:y + h+img_space, x-img_space:x + w+img_space]

        img_item = "face\\{0}.png".format(str(i))
        print(img_item)
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x-img_space, y-img_space), (end_cord_x+img_space, end_cord_y+img_space), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
