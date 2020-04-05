from dataset import Dataset
from histogram import histogram


def main():
    dataset = Dataset()
    all_images, all_image_names = dataset.get_all_images()
    his = [histogram(i, 16) for i in all_images]
    pass


if __name__ == '__main__':
    main()
