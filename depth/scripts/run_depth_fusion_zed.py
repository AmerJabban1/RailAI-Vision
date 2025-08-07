# scripts/run_depth_fusion_zed.py

"""
Real-time detection + segmentation + depth measurement using ZED2 camera.
"""

import cv2
import torch
import pyzed.sl as sl
from depth.models.loader import load_yolo, load_deeplab
from depth.engine import fuse_outputs
from depth.utils.visualization import draw_detection, overlay_segmentation_mask

# ----------------------------
# Configuration
# ----------------------------
CONF_THRESHOLD = 0.4
TRACK_CLASS_ID = 12  # Rail class ID in RailSem19
CLASS_NAMES = ["Person", "Traffic Sign/Signal", "Vehicle", "Obstacle", "Rail"]

# ----------------------------
# ZED Camera Initialization
# ----------------------------
def initialize_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.PERFORMANCE,
        coordinate_units=sl.UNIT.METER
    )
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Failed to open ZED2 camera")
    return zed, sl.RuntimeParameters()

# ----------------------------
# Main
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = load_yolo().to(device)
    seg_model = load_deeplab().to(device).eval()

    zed, runtime_params = initialize_zed()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    print("[INFO] Real-time ZED2 fusion running. Press 'q' to quit.")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

            frame = image_zed.get_data()
            depth_map = depth_zed.get_data()
            bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            fused_img = fuse_outputs(
                image=bgr_image,
                depth_map=depth_map,
                yolo_model=yolo_model,
                seg_model=seg_model,
                device=device,
                class_names=CLASS_NAMES,
                track_class_id=TRACK_CLASS_ID,
                conf_threshold=CONF_THRESHOLD,
                calibrate_depth=False  # ZED already calibrated
            )

            cv2.imshow("RailAI-Vision | ZED2 Depth Fusion", fused_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()