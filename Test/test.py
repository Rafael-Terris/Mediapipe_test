import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 这里确认你的模型路径是正确的
model_path = '/Users/Terrious/Desktop/Learning/Self/Mediapipe_test/Test/face_landmarker.task'

# 设置一个全局变量，用于接收异步回调传回来的识别结果
latest_result = None


# 这是回调函数：当模型在一帧画面里识别完人脸后，会自动调用这个函数
def update_result(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


# 配置检测器参数
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result
)

# 1. 打开你的电脑摄像头（0 通常代表默认摄像头）
cap = cv2.VideoCapture(0)

# 2. 创建检测器实例
with FaceLandmarker.create_from_options(options) as landmarker:
    # 3. 写一个死循环，不断从摄像头读取画面
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("未能读取到摄像头画面")
            break

        # 4. 获取当前画面的时间戳（毫秒级，LIVE_STREAM 模式必须提供）
        frame_timestamp_ms = int(time.time() * 1000)

        # 5. OpenCV 默认是 BGR 格式，而 MediaPipe 需要 RGB 格式，所以转换一下颜色
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 6. 把 numpy 数组转换成 MediaPipe 认识的 Image 对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 7. 把图像和时间戳送进检测器进行异步检测
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # 8. 如果回调函数抓到了人脸结果，我们在画面左上角写点字，并输出特征点坐标
        if latest_result and latest_result.face_landmarks:
            faces_count = len(latest_result.face_landmarks)
            cv2.putText(frame, f"Faces detected: {faces_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 遍历画面中检测到的每一张脸
            for face_idx, face_landmarks in enumerate(latest_result.face_landmarks):
                # 获取画面当前的宽度和高度
                h, w, _ = frame.shape

                # 提取第 1 号特征点（鼻尖）
                nose_tip = face_landmarks[1]

                # MediaPipe 默认输出的是 0.0 到 1.0 的“归一化坐标”（比例）
                norm_x = nose_tip.x
                norm_y = nose_tip.y
                norm_z = nose_tip.z

                # 如果你想得到它在画面上的实际像素坐标，只需要乘上宽和高
                pixel_x = int(norm_x * w)
                pixel_y = int(norm_y * h)

                # 打印到控制台
                print(f"Face {face_idx} | 鼻尖归一化坐标: x={norm_x:.3f}, y={norm_y:.3f}")
                print(f"Face {face_idx} | 鼻尖实际像素坐标: X={pixel_x}, Y={pixel_y}")

                # 我们甚至可以直接在鼻尖上画个红色的圆圈来验证坐标对不对
                cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 0, 255), -1)

        # 9. 实时显示画面
        cv2.imshow('MediaPipe Face Landmarker', frame)

        # 10. 如果按下键盘上的 'q' 键，就退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()