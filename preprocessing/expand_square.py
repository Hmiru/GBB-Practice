from torchvision.transforms import Pad

class ExpandSquare(object):
    def __call__(self, image):
        original_width, original_height = image.size

        if original_width == original_height:
            return image
        elif original_width > original_height:
            diff = original_width - original_height
            pad_tuple = (0, diff // 2) if diff % 2 == 0 else (0, diff // 2, 0, diff // 2 + 1)
        else:
            diff = original_height - original_width
            pad_tuple = (diff // 2, 0) if diff % 2 == 0 else (diff // 2, 0, diff // 2 + 1, 0)

        return Pad(pad_tuple)(image)