import torch
import torchvision.datasets as dsets

BATCH_SIZE = 128


def main():
    train_dataset = dsets.CelebA(root='/datashare/',
                                 split='train',
                                 download=False)  # make sure you set it to False

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)


if __name__ == '__main__':
    main()
