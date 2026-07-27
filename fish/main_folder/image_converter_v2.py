import cv2
from numpy import ndarray
from typing import Optional


class ImageConverter:
    def __init__(
            self,
            original_shape: tuple[int, int] = (),
            converted_shape: tuple[int,int] = (),
            padding_color: Optional[tuple[int, int, int]] = (114, 114, 114)
    ):
        self._padding_color: tuple[int, int, int] = padding_color
        self._shape_checked: bool = False
        self._width, self._height = original_shape
        self._new_width, self._new_height = converted_shape
        self._scale: float = min(self._new_width / self._width, self._new_height / self._height)
        self._resized_width: int = int(round(self._width * self._scale))
        self._resized_height: int = int(round(self._height * self._scale))
        self._padding_width: int = self._new_width - self._resized_width
        self._padding_height: int = self._new_height - self._resized_height
        self._top: int = int(round(self._padding_height / 2 - 0.1))
        self._bottom: int = int(round(self._padding_height / 2 + 0.1))
        self._left: int = int(round(self._padding_width / 2 - 0.1))
        self._right: int = int(round(self._padding_width / 2 + 0.1))

    def letterbox(self, image: ndarray):
        """Пропорционально изменяет размер изображения и заполняет серым цветом недостающие пиксели."""
        if not self._shape_checked:
            self._shape_checked = True

            if not self._is_image_shape_same(image):
                print(f"Размер изображения не соответствует размеру, заданному при инициализации!")
                # TODO: self._shape_can_be_dynamic реализовать позже
                return

        image = cv2.resize(
            image,
            (self._resized_width, self._resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        image = cv2.copyMakeBorder(
            image,
            self._top,
            self._bottom,
            self._left,
            self._right,
            cv2.BORDER_CONSTANT,
            value=self._padding_color,
        )

        return image

    def calculate_reverse_coordinates(self, prediction: list) -> tuple[int, int, int, int]:
        """Высчитывает координаты объекта на оригинальном размере изображения."""
        x1, y1, x2, y2, *_ = prediction

        new_x1 = (x1 - self._left) / self._scale
        new_y1 = (y1 - self._top) / self._scale
        new_x2 = (x2 - self._left) / self._scale
        new_y2 = (y2 - self._top) / self._scale

        return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

    def _is_image_shape_same(self, image: ndarray):
        """Проверяет что изображение соответствует размеру заданному при инициализации."""
        height, width = image.shape[:2]
        return bool(height == self._height and width == self._width)
