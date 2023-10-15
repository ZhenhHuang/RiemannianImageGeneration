import torch
from torchvision.transforms.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader


datasize = {
    'MNIST': {"size": (1, 28, 28), "in_dim": 28 * 28},
    'USPS': {"size": (1, 16, 16), "in_dim": 16 * 16},
    'cifar-10': {"size": (3, 32, 32), "in_dim": 3 * 32 * 32},
    'cifar-100': {"size": (3, 32, 32), "in_dim": 3 * 32 * 32}
}


class Caltech101(Dataset):
    def __init__(self, root_path='./datasets', data_path='caltech101', flag='train', size=None, transform=None):
        super(Caltech101, self).__init__()
        assert flag in ['train', 'val', 'test', 'pred']
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        if size is None:
            self.height = 288
            self.width = 288
        else:
            self.height = size[0]
            self.width = size[1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation((-90, 90), interpolation=InterpolationMode.BILINEAR),
                transforms.Resize((self.height, self.width)),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])
        if flag != 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.height, self.width)),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        self.__read_data__()

    def __getitem__(self, index):
        data = Image.open(self.data[index], mode='r')
        data = self.transform(data.convert('RGB'))
        if self.flag != 'pred':
            label = self.label[index]
            return data, torch.tensor(label)
        return data

    def __len__(self):
        return len(self.data)

    def __read_data__(self):
        src = f"{self.root_path}/dataset.zip"
        dst = self.root_path

        if not os.path.exists(f"{self.root_path}/{self.data_path}"):
            unzip(src, dst)

        path = os.path.join(self.root_path, self.data_path)
        self.map_dict = self.__getClassMap(path)
        target_path = os.path.join(path, "test.txt") if self.flag == 'pred' else os.path.join(path, "train.txt")

        f = open(target_path)
        data = []
        label = []
        self.data_names = []
        lines = f.readlines()
        f.close()

        if self.flag != "pred":
            mapping = {'train': 0, 'val': 1, 'test': 2}
            n = len(lines)
            train = int(n * 0.8)
            val = int(n * 0.1)
            border1 = [0, train, train + val]
            border2 = [train, train + val, n]
            index1 = border1[mapping[self.flag]]
            index2 = border2[mapping[self.flag]]
            lines = lines[index1: index2]

        for line in lines:
            if self.flag != 'pred':
                pic_name, cls = line.split()
                label.append(int(cls))
            else:
                pic_name = line.strip()
                self.data_names.append(pic_name.split(".")[0])
            data.append(f'{path}/images/{pic_name}')

        self.data = data
        self.label = np.array(label)

    def __getCountDict(self):
        self.count_dict = {}
        for i in self.label:
            self.count_dict[i] = self.count_dict.get(i, 0) + 1

    def __getClassMap(self, path):
        f = open(f"{path}/class.txt")
        map_dict = {}
        for line in f.readlines():
            name, id = line.split()
            map_dict[name] = int(id)
        f.close()
        return map_dict