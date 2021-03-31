from . import FingerprintImageEnhancer
import cv2

def basicEnhancing(img):
    ret,img = cv2.threshold(img,140,255,cv2.THRESH_TOZERO)

    img = cv2.medianBlur(img,5)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,43,1)

    # Normalize and threshold image
    im = cv2.normalize(th3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    res, im = cv2.threshold(im, 64, 255, cv2.THRESH_BINARY)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im, None, (0,0), 0)
    # cv2.imwrite('my_o1.png', im)
    return im

def advancedEnhancing(img):
    image_enhancer = FingerprintImageEnhancer.FingerprintImageEnhancer()         # Create object called image_enhancer
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out = image_enhancer.enhance(img)     # run image enhancer
    out_image = 255 - (255 * out)
    return out_image

    # image_enhancer.save_enhanced_image('../enhanced/' + img_name)   # save output
