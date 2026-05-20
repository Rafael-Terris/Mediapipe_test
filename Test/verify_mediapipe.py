import cv2
import mediapipe as mp

def main():
    print("正在初始化 MediaPipe...")
    # 使用基础的解决方案 API，无需额外下载模型文件，最适合快速验证
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 开启默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查设备连接或权限。")
        return

    print("摄像头已打开！请看向摄像头。按键盘上的 'ESC' 键退出。")

    # 配置 FaceMesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("无法读取摄像头画面。")
                break

            # 处理图像，转换为 RGB (MediaPipe 需要 RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            # 如果检测到人脸，绘制特征点
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

            # 显示结果，水平翻转（镜像）
            cv2.imshow('MediaPipe Verification', cv2.flip(image, 1))

            # 监听按键，按 ESC 退出
            if cv2.waitKey(5) & 0xFF == 27:
                print("正在退出...")
                break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("验证脚本运行结束。")

if __name__ == "__main__":
    main()
