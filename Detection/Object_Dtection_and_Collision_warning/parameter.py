model_name = "yolov8m.pt"
distance_parameter = 0.2
image_size = 640
confidence = 0.3
selected_class = (1, 2, 3, 5, 7)

def distance_calculator(normalized_width):
    distance = round((1 - normalized_width) ** 4, 1)
    return distance