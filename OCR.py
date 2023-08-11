import logging
from paddleocr import PaddleOCR
import cv2
import numpy as np
import pandas as pd
from urllib import request
from tqdm import tqdm


def text_recognition(img):
    logging.disable(logging.DEBUG)
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    result = ocr.ocr(img, cls=True)
    txts = [line[1][0] for line in result[0]]
    return txts

def image_process(img, center):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    w = 0.15 * 255
    img = img[::-1, :].copy()
    img = image_cutting(img, scan_width=50, rate=0.00001, center=center, w=w, margin=[40, 40])
        # plt.imshow(img)
        # plt.show()
    img = img[::-1, :]
    img = color_slicing(img, center, w)
    return img

# 裁剪图像
def image_cutting(image, scan_width, rate, center, w, margin):
    x_upper, y_upper = image.shape[:2]

    total = y_upper * scan_width
    for x in range(margin[1], x_upper - scan_width):
        subimage = image[x : x + scan_width, :]
        count = np.sum(
            np.logical_and.reduce(
                np.abs(subimage - center) < (w / 2),
                axis=2
            )
        )
        if count / total < rate:
            x_upper = x + scan_width // 2
            break

    total = x_upper * scan_width
    for y in range(margin[0], y_upper - scan_width):
        subimage = image[:, y : y + scan_width]
        count = np.sum(
            np.logical_and.reduce(
                np.abs(subimage - center) < (w / 2),
                axis=2
            )
        )
        if count / total < rate:
            y_upper = y + scan_width // 2
            break

    return image[margin[1]:x_upper, margin[0]:y_upper]

# 二值化图像
def color_slicing(image, center, w):
    """
    :param image:
    :param center: b, g, r ib range 0 ~ 255
    :param w: width
    :return:
    """
    lower_bound = np.array(center) - w / 2
    upper_bound = np.array(center) + w / 2 

    mask = np.all((image >= lower_bound) & (image <= upper_bound), axis=-1)
    out = np.zeros_like(image)
    out[mask] = (255, 255, 255)

    return out

# 检测地址字段
def mandarin(txt):
    out = ''
    for word in txt:
        for char in word:
            if  '\u4e00' <= char <= '\u9fff':
                out = out + word
                break
    return out

def read_data(excel_name, sheet_name, content_col, n_rows=-1):
    if n_rows < 0:
        data = pd.read_excel(excel_name, sheet_name, usecols=content_col).dropna()
    else:
        data = pd.read_excel(excel_name, sheet_name, usecols=content_col, nrows= n_rows).dropna()
    return data

def image_getting(URL):
    try:
        res = request.urlopen(URL)
        img = np.asarray(bytearray(res.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        res.close()
    except(ConnectionResetError,TimeoutError):
        image_getting(URL)
    return img

if __name__ == '__main__':
    table = read_data(excel_name='自营延保USS2.0完工照片链接.xlsx', # 读取的excel
              sheet_name='自营延保5-6月完工照片', # 读取的sheet
              content_col=['图片地址', '工单号', '用户地址'], # 读取的列
              n_rows=10, # 读取行数，全部读取设为-1
              )
    
    center = (255, 255, 255) #水印颜色
    total_rows = len(table)
    txts = []
    locs = []
    for index, row in tqdm(table.iterrows(), total=total_rows, desc="处理中："):
        url = row['图片地址']
        img = image_getting(url)
        img = image_process(img, center)
        txt = text_recognition(img)
        loc = mandarin(txt)
        txts.append(txt)
        locs.append(loc)

    table['识别的所有文字'] = txts
    table['识别地址'] = locs 
    table.to_excel(excel_writer='OCR.xlsx', sheet_name='识别结果', index=False)