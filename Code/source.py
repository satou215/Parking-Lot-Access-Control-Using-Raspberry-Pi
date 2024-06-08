import cv2
from PIL import Image
import mysql.connector
import datetime
import numpy as np
import math
import time
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import board        
import neopixel
from gpiozero import LED

green = LED(27)
red = LED(17)
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) #kích cỡ càng to thì càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

n = 1

Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

#Hàm mở kết nối Mysql với database tên "inoutcontrol"
def connectDB():
    con = mysql.connector.connect(
        host = 'localhost',
        user = 'root',
        password = '21521913',
        database = 'inoutcontrol'
	)
    return con

#Check tên biển số đọc từ hình ảnh lưu vào thue mục "images" đã tồn tại trong database chưa
def checkNp(number_plate):
    con = connectDB()
    cursor = con.cursor()
    sql = "SELECT * FROM licenseplate WHERE number_plate = %s"
    cursor.execute(sql, (number_plate,))
    cursor.fetchall()
    result = cursor._rowcount
    con.close()
    cursor.close()
    return result

# Check tên biển số và trạng thái của bản ghi gần nhất đọc từ hình ảnh lưu vào thư mục images
def checkNpStatus(number_plate):
    con = connectDB()
    cursor = con.cursor()
    sql = "SELECT * FROM licenseplate WHERE number_plate = %s ORDER BY date_in DESC LIMIT 1"
    cursor.execute(sql, (number_plate,))
    result = cursor.fetchone()
    con.close()
    cursor.close()
    return result

#Tạo bản ghi dành cho xe vào bãi giữ xe (Cho xe vào bãi)
#TH1: Tên biển số xe đọc từ ảnh chưa tồn tại trong database
#TH2: Tên biển số xe đọc từ ảnh đã tồn tại trong database

def insertNp(number_plate, key):
    con = connectDB()
    cursor = con.cursor()
    sql = "SELECT COUNT(*) FROM licenseplate"
    cursor.execute(sql)
    result = cursor.fetchone()
    num_result = ''
    char_result = str(result)
    for c in char_result:
        if c.isnumeric():
            num_result = num_result + c;
    char_result = str(int(num_result) + 1)
    sql = "INSERT INTO licenseplate(ID, number_plate, status, date_in, RFID) VALUES(%s,%s,%s,%s, %s)"
    now = datetime.datetime.now()
    date_in = now.strftime("%Y/%m/%d %H:%M:%S")
    key_str = str(key)
    cursor.execute(sql, (char_result,number_plate, '1', date_in, key_str))
    con.commit()
    cursor.close()
    con.close()
    print('VAO BAI GUI XE')
    print('Ngay gio vao: ' + datetime.datetime.strftime(datetime.datetime.now(), "%Y/%m/%d %H:%M:%S"))

#Cập nhật bản ghi (Cho xe ra khỏi bãi)
def updateNp(Id):
    con = connectDB()
    cursor = con.cursor()
    sql = "UPDATE licenseplate SET status = 0, date_out = %s WHERE Id = %s"
    now = datetime.datetime.now()
    date_out = now.strftime("%Y/%m/%d %H:%M:%S")
    cursor.execute(sql, (date_out, Id))
    con.commit()
    cursor.close()
    con.close()
    print('RA KHOI BAI GUI XE')
    print('Ngay gio ra: ' + datetime.datetime.strftime(datetime.datetime.now(), "%Y/%m/%d %H:%M:%S"))

def datain(number_plate, key):
    if(number_plate != ''):
        #Gọi hàm kiểm tra xem biển số đã tồn tại trong database chưa
        check = checkNp(number_plate)
        if(check == 0):
            #Nếu xe chưa từng đến gửi tại bãi thì gọi hàm insertNp để cho xe vào gửi
            insertNp(number_plate, key)
            led_accepted()
        else:
            #Gọi hàm kiểm tra trạng thái của xe
            check2 = checkNpStatus(number_plate)
            #Nếu trạng thái của xe = 1(xe vào gửi và chưa lấy ra)
            if(check2[2] == 1):
                 #Gọi hàm updateNp lấy xe ra và cập nhật trạng thái cho xe = 0
                if(check2[5] == str(key)):
                    updateNp(check2[0])
                    led_accepted()
                else:
                    print("Sai the")
                    led_denied()
                    
            #Nếu trạng thái của xe  = 0 (xe vào gửi và lấy ra rồi)
            else:
                #Gọi hàm insertNp để cho xe vào gửi
                insertNp(number_plate, key)
                led_accepted()
    else:
        print('Bien so khong xac dinh')
        led_denied()

def preprocess(imgOriginal):

    imgGrayscale = extractValue(imgOriginal)
    # Trả về giá trị cường độ sáng, hàm extractValue chỉ trả về giá trị điểm ảnh, không có giá trị màu ==> ảnh gray
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale) #để làm nổi bật biển số hơn, dễ tách khỏi nền
    height, width = imgGrayscale.shape # Lấy ra giá trị height và width của hình ảnh

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    #tạo một hình ảnh trắng hoặc đen có kích thước height x width pixels với một kênh màu duy nhất (grayscale) và kiểu dữ liệu (dtype) là uint8.
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    #Làm mịn ảnh bằng bộ lọc Gauss 5x5, sigma = 0

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    #Tạo ảnh nhị phân

    return imgGrayscale, imgThresh
    #Trả về ảnh xám và ảnh nhị phân
# end function

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    #Lấy ra chiều cao, chiều rộng và số kênh màu
    imgHSV = np.zeros((height, width, 3), np.uint8)
    #tạo một mảng trắng (tất cả giá trị là 0) với kích thước (height, width, 3) và kiểu dữ liệu là uint8 (unsigned 8-bit integer)
    #được sử dụng để lưu trữ hình ảnh trong không gian màu HSV
    
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    #chuyển đổi hình ảnh ban đầu từ không gian màu BGR (Blue, Green, Red) sang không gian màu HSV (Hue, Saturation, Value)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    #màu sắc, độ bão hòa, giá trị cường độ sáng
    #Không chọn màu RBG vì vd ảnh màu đỏ sẽ còn lẫn các màu khác nữa nên khó xác định ra "một màu" 

    return imgValue
# end function


def check_firstline(first_line):
    if(len(first_line) != 4):
            return False
    if len(first_line) == 4:
        if(first_line[2].isalpha() == False or first_line[0].isnumeric() == False or first_line[1].isnumeric() == False or first_line[3].isnumeric() == False):
            return False
    return True
def check_secondline(second_line):
    if len(second_line) != 5:
        return False
    if second_line.isnumeric():
        return True;
    return False
###################################################################################################
def maximizeContrast(imgGrayscale):
    #Làm cho độ tương phản lớn nhất 
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat) 
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    #Kết quả cuối là ảnh đã tăng độ tương phản 
    return imgGrayscalePlusTopHatMinusBlackHat

# end function



#Đọc biển số xe lưu ở "images"
def readnumberplate(key):
    img = cv2.imread("/home/duc3503/LuongDai/images/numberplate.jpg")
    number_plate = ''
    npaClassifications = np.loadtxt("/home/duc3503/LuongDai/classifications.txt", np.float32)
    npaFlattenedImages = np.loadtxt("/home/duc3503/LuongDai/flattened_images.txt", np.float32)
    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
    kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    #########################

    ################ Image Preprocessing #################
    imgGrayscaleplate, imgThreshplate = preprocess(img)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation

    ###########################################

    ###### Draw contour and filter out the license plate  #############
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất
   
    listNp = {}
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tính chu vi
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h

        if (len(approx) == 4):
            screenCnt.append(approx)

            cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

    if screenCnt is None:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:

        for screenCnt in screenCnt:
            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

            ############## Find the angle of the license plate #####################
            (x1, y1) = screenCnt[0, 0]
            (x2, y2) = screenCnt[1, 0]
            (x3, y3) = screenCnt[2, 0]
            (x4, y4) = screenCnt[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            sorted_array = array.sort(reverse=True, key=lambda x: x[1])
            (x1, y1) = array[0]
            (x2, y2) = array[1]
            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = 0
            if ke != 0:
                angle = math.atan(doi / ke) * (180.0 / math.pi)
            ####################################

            ########## Crop out the license plate and align it to the right angle ################

            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

            # Cropping
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx, topy:bottomy]
            imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

            ####################################

            #################### Prepocessing and Character segmentation ####################
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số

            ##################### Filter out characters #################
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width

            for ind, cnt in enumerate(cont):
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                char_area = w * h

                if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind


            ############ Character recognition ##########################

            char_x = sorted(char_x)
            first_line = ""
            second_line = ""

            for i in char_x:
                (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                imgROI = thre_mor[y:y + h, x:x + w]  # Crop the characters
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image
                npaROIResized = imgROIResized.reshape(
                    (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

                npaROIResized = np.float32(npaROIResized)
                _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # call KNN function find_nearest;
                strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
                cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
                if (y < height / 3):  # decide 1 or 2-line license plate
                    first_line = first_line + strCurrentChar
                else:
                    second_line = second_line + strCurrentChar
            listNp[first_line] = second_line
            roi = cv2.resize(roi, None, fx=0.75, fy=0.75)

    for x in listNp:
        first_line = x
        second_line = listNp[x]
        if check_firstline(first_line) and check_secondline(second_line):
            print("\n License Plate is: " + first_line + " - " + second_line + "\n")
            number_plate = first_line + " - " + second_line
            datain(number_plate,key)
            return
    print("Khong tim thay bien so xe\n")
    led_denied()

def led_accepted():
    green.on()
    time.sleep(2)
    green.off()
   

def led_denied():
        
    red.on()
    time.sleep()
    red.off()



reader = SimpleMFRC522()
cam = cv2.VideoCapture(0)

while True:
    #Đọc từng frame
    ret,frame=cam.read()
    #Thiết lập màu sắc cho ảnh
    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    key, text = reader.read()

    if key == 157502735209 or key == 40551782013 or key == 764270878022 or key == 762480893260:
        #Lưu ảnh vào thư mục images
        cv2.imwrite('/home/duc3503/LuongDai/images/numberplate.jpg', framegray)
        #Gọi hàm xử lý "readnumberplate"
        readnumberplate(key)
    else:
        led_denied()
    time.sleep(1)

cam.release()
cv2.destroyAllWindows()


