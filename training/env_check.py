import importlib.util

def check_package_installed(package_name):
    package_spec = importlib.util.find_spec(package_name)
    if package_spec is not None:
        print(f"{package_name} is installed.")
    else:
        print(f"{package_name} is NOT installed.")

# Check if the required libraries are installed
check_package_installed('imutils')
check_package_installed('cv2')  # OpenCV package
check_package_installed('deepface')
check_package_installed('mediapipe')
