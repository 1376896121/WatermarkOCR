# OCRforWatermark
This is a simple work for my internship in TCL, 2023 August.
可以实现从图片提取位于不同位置、不同颜色的水印。

## 安装依赖：
建议使用python3.8-3.9<br>
`pip install paddleocr paddlepaddle`

## 使用方法：
在OCR.py中更改以下位置：<br>
107行：设置输入路径<br>
128行：设置输出路径<br>
17行：image_process中，可以更改水印位置<br>
113行：更改水印颜色<br>
执行：`python OCR.py`
