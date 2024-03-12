import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

selfie = mp.solutions.selfie_segmentation

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

# Create a image segmenter instance with the live stream mode:
def print_result(result, output_image, timestamp_ms):
    # cv2.imshow("output", output_image)
    # print('segmented masks size: {}'.format(len(result)))
    print(timestamp_ms)
    # cv2.imshow('Show', output_image.numpy_view())

if __name__ == '__main__':
    bxs=np.full((3,3,2),0)
    bxl=np.full((3,3,2),0)
    si=600
    bl=[0,si//3,(si//3)*2,si]
    for i in range(3):
        for j in range(3):
            bxs[i,j]=(bl[i],bl[j])
            bxl[i,j]=(bl[i+1],bl[j+1])

    # Capture live video
    cap = cv2.VideoCapture(0)

    # Create the options that will be used for ImageSegmenter
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_category_mask=True,
        result_callback=print_result)


    with ImageSegmenter.create_from_options(options) as segmenter:

        # initialize the Hands class and store it in a variable
        # cmp_hands = mp.solutions.hands

        # Set hands function which will hold the landmarks points
        # hands = mp_hands.Hands(static_image_mode=True)

        # Drawing function of hand landmarks on the image
        # mp_drawing = mp.solutions.drawing_utils

        tms =0

        while cap.isOpened():
            # Capture and process image frame from video
            success, image = cap.read()
            # print(success)
            image=cv2.flip(image,1)
            image = cv2.resize(image,(si+200,si), fx = 0.1, fy = 0.1)
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            tms += 1

            # results = hands.process(imageRGB)
            # ck=0

            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            # Retrieve the masks for the segmented image
            # segmentation_result = segmenter.segment_async(mp_image, 100)
            segmenter.segment_async(mp_image, tms)
            # category_mask = segmentation_result.category_mask

            cv2.imshow("output", image)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        cv2.destroyAllWindows()
        cap.release()
