import cv2
from typing import Optional


class ImageConverter:
    def __init__(self, original_img = None, converted_shape: tuple = (), padding_color: Optional[tuple] = ()):
        self.converted_shape = converted_shape
        self.image = original_img
        self.padding_color = padding_color
        self.scale = 0.0
        self.padding_height = 0.0
        self.padding_width = 0.0

    def letterbox(self):
        """Делает ресайз изображения и заполняет серым цветом недостающие пиксели, как это делала библиотека YOLO при обучении модели."""
        image = self.image
        height, width = self.image.shape[:2]
        new_width, new_height = self.converted_shape

        self.scale = min(new_width / width, new_height / height)

        resized_width = int(round(width * self.scale))
        resized_height = int(round(height * self.scale))

        self.padding_width = new_width - resized_width
        self.padding_height = new_height - resized_height

        top = int(round(self.padding_height / 2 - 0.1))
        bottom = int(round(self.padding_height / 2 + 0.1))
        left = int(round(self.padding_width / 2 - 0.1))
        right = int(round(self.padding_width / 2 + 0.1))

        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.padding_color,
        )

        return image

    def calculate_reverse_coordinates(self, prediction: list) -> tuple[int, int, int, int]:
        x1, y1, x2, y2, *_ = prediction

        left = self.padding_width / 2
        top = self.padding_height / 2

        x1 = (x1 - left) / self.scale
        y1 = (y1 - top) / self.scale
        x2 = (x2 - left) / self.scale
        y2 = (y2 - top) / self.scale

        return int(x1), int(y1), int(x2), int(y2)