import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

"""
    Không dùng ImageFolder nhé
"""
class AnimalDataset(Dataset):
    def __init__(self, root, train=True):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        image = None
        label = None
        return image, label


"""
    Giới thiệu bộ dataset football
    - Mỗi 1 video kéo dài khoảng 1 phút, với 25 khung hình 1 giây (25 FPS)
    - Mỗi 1 video sẽ có 1 file annotation .json tương ứng (Mình khuyến khích các bạn mở các file này ra và
    thử cố gắng hiểu về các attribute trong các file này nhé)
    - Đối với từ khóa "categories": Nhìn chung sẽ có 4 đối tượng được annotate trong các video, với ID là 1 cho đến 4. Tạm thời các bạn
    chỉ cần quan tâm đến id = 4, là các cầu thủ là được
    - Đối với từ khóa "images": Đây là thông tin về các frame trong video. Các bạn chú ý là ở đây frame xuất phát từ 1,
    nhưng trong lập trình chỉ số xuất phát từ 0 nhé. Nhìn chung sẽ có 1500 frames, tương ứng với 1 phút
    - Đối với từ khóa "annotations": ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT. Các bạn sẽ thấy trong trường này có rất nhiều
    dictionary, mỗi 1 dictionary tương ứng với 1 object trong 1 frame nhất định, trong đó:
        + id: không cần quan tâm
        + image_id: id của frame (chạy từ 1 cho đến 1500)
        + category_id: Các bạn chỉ cần quan tâm đến những item mà category_id = 3 (player) là được
        
    TASK: Các bạn hãy xây dựng Dataset cho bộ dataset này, với các quy tắc sau:
    - Hàm __init__ tùy ý các bạn thiết kế
    - Hàm __len__ trả về tổng số lượng frame có trong tất cả các video
    - Hàm __getitem__(self, idx) trả về list của các bức ảnh đã được crop về các cầu thủ (trong hầu hết các frame
    là sẽ có 1 cầu thủ trong 1 frame) và list các số áo tương ứng của các cầu thủ này. idx sẽ theo quy tắc sau: Giả sử
    các bạn gộp tất cả các video thành 1 video dài (thứ tự các video con tùy các bạn), thì idx sẽ là index của video 
    dài đó. Ví dụ trong trường hợp chúng ta có 3 video con dài 1 phút, thì video tổng sẽ dài khoảng 3 phút và có
    4500 frames tổng cộng.
    
    GOOD LUCK!
"""
class FootballDataset(Dataset):
    def __init__(self, root):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        images = [None for _ in range(10)]
        labels = [0 for _ in range(10)]
        return images, labels

class CIFARDataset(Dataset):
    def __init__(self, root=".", train=True, transform=None):
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item].reshape((3, 32, 32)).astype(np.float32)
        if self.transform:
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        label = self.labels[item]
        return image, label


if __name__ == '__main__':
    dataset = CIFARDataset(root="../data")
    index = 400
    image, label = dataset.__getitem__(index)
    print(image.shape)
    print(label)
