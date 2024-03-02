import cv2
from dlib import get_frontal_face_detector, shape_predictor
import numpy as np
import matplotlib.pyplot as plt

def get_image_size(image):
    return (image.shape[0], image.shape[1])

def get_landmarks(image, detector, predictor):
    dets = detector(image, 1)
    if len(dets) == 0:
        raise IndexError('dets is none.')
    shape = predictor(image, dets[0])
    return np.array([[part.x, part.y] for part in shape.parts()])

def get_mask(image_size, landmarks):
    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([landmarks[0:16], landmarks[26:17:-1]])
    cv2.fillPoly(img=mask, pts=[points], color=255)
    return mask

def affine_trans(origin_im, target_im, origin_landmarks, target_landmarks):
    anchor_points = [18, 8, 25]
    M = cv2.getAffineTransform(origin_landmarks[anchor_points].astype(np.float32),
                               target_landmarks[anchor_points].astype(np.float32))
    return cv2.warpAffine(origin_im, M, (target_im.shape[1], target_im.shape[0]))

def union(target_mask, affine_origin_mask):
    mask = np.min([target_mask, affine_origin_mask], axis=0) 
    mask = ((cv2.blur(mask, (5, 5)) == 255) * 255).astype(np.uint8)
    mask = cv2.blur(mask, (3, 3)).astype(np.uint8)
    return mask

def skin_color_adjustment(affine_origin_im, target_im):
    im1_ksize = 77
    im2_ksize = 77
    im1_factor = cv2.GaussianBlur(affine_origin_im, (im1_ksize, im1_ksize), 0).astype(np.float)
    im2_factor = cv2.GaussianBlur(target_im, (im2_ksize, im2_ksize), 0).astype(np.float)

    affine_origin_im = np.clip((affine_origin_im.astype(np.float64) * im2_factor / np.clip(im1_factor, 1e-6, None)), 0, 255).astype(np.uint8)
    return affine_origin_im

def center_point(image):
    image_index = np.argwhere(image > 0)
    miny, minx = np.min(image_index, axis=0)
    maxy, maxx = np.max(image_index, axis=0)
    center_point = ((maxx + minx) // 2, (maxy + miny) // 2)
    return center_point

def show(img):
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

def init(target_image):
    """
    Initialization is performed for the fusion requirements.
    It can be initialized ONLY ONCE. 
    Used to initialize the detector, load the model file(.dat) and generate the face mask.
    :param target_image: target or base image is used to fuse.
    :return detector: get_frontal_face_detector().
            predictor: load model file(shape_predictor_68_face_landmarks.dat).
            target_im: load target image.
            target_landmarks: detecting facial landmarks for target image.
            target_mask: generating mask for target image.
    """
    detector = get_frontal_face_detector()
    predictor = shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    target_im = cv2.imread(target_image)
    target_landmarks = get_landmarks(target_im, detector, predictor)
    target_mask = get_mask(get_image_size(target_im), target_landmarks)
    return detector, predictor, target_im, target_landmarks, target_mask

def fuse(origin_im, detector, predictor, target_im, target_landmarks, target_mask):
    """
    See the following steps for face swap:
    1. Extracting the 68 landmarks from origin image. 
    2. Generating the face mask according to landmarks.
    3. Aligning the face between origin and target image by affine transformation.
    4. Generating affine origin image and affine origin mask depends on step 3.
    5. Taking the union of target and affine origin mask.
    6. Fusing the affine origin image and target by Poisson fusion in opencv.
    :param origin_im:
           detector: get_frontal_face_detector().
           predictor: model file(shape_predictor_68_face_landmarks.dat).
           target_im: target image
           target_landmarks: target image landmarks.
           target_mask: target image mask. 
    :return seamless_im: result image.
    """
    try:
        origin_landmarks = get_landmarks(origin_im, detector, predictor)
    except IndexError:
        return target_im
    origin_mask = get_mask(get_image_size(origin_im), origin_landmarks)
    affine_origin_im = affine_trans(origin_im, target_im, origin_landmarks, target_landmarks)
    affine_origin_mask = affine_trans(origin_mask, target_im, origin_landmarks, target_landmarks)
    union_mask = union(target_mask, affine_origin_mask)
    affine_origin_im = skin_color_adjustment(affine_origin_im, target_im)

    seamless_im = cv2.seamlessClone(affine_origin_im, target_im, 
                                    mask=union_mask, 
                                    p=center_point(affine_origin_mask), 
                                    flags=cv2.NORMAL_CLONE)
    return seamless_im
