<div align="center">
<img alt="" src="https://github-production-user-asset-6210df.s3.amazonaws.com/125717930/246971879-8ce757c3-90dc-438d-807f-3f3d29ddc064.png" width=500/>
</div>


<div align="left">
<img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg" alt="Awesome Badge"/>
<img src="https://img.shields.io/static/v1?label=%F0%9F%8C%9F&message=If%20Useful&style=style=flat&color=BC4E99" alt="Star Badge"/>
<img src="https://img.shields.io/github/issues/genderev/assassin" alt="issue"/>
<img src="https://img.shields.io/github/issues-pr/genderev/assassin" alt="pr"/>
</div>

<details open>
<summary><h2>How it works</h2></summary>


https://user-images.githubusercontent.com/82228271/210924923-b115c6c2-c748-4fba-abd1-1843503a98bf.mp4


</details>
<details open>
<summary><h2>Installation</h2></summary>
  
```
pip install kbyocr
```
 
</details>
<details open>
<summary><h2>Documentation</h2></summary>

<h3>Denoise OCR</h3>

```
import cv2
from kbyocr import denoise_ocr

image = cv2.imread('test.png')
result = denoise_ocr(image)
cv2.imwrite('result.png', result)
```

<h3>Thinning</h3>

```
import cv2
from kbyocr import thinning

image = cv2.imread('unit_test/result.png')
result = thinning(image)
cv2.imwrite('unit_test/thin.png', result)
```

<h3>Binarization</h3>

```
import cv2
from kbyocr import binarization

image = cv2.imread('unit_test/result.png')
result = binarization(image)
cv2.imwrite('unit_test/unit_test/binary.png.png', result)
```

</details>

